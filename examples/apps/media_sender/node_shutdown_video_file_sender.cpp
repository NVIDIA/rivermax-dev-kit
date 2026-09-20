/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <chrono>
#include <memory>
#include <thread>
#include <utility>

#include "rdk/apps/media_sender/media_sender.h"
#include "rdk/examples/apps/media_sender/node_shutdown_video_file_sender.h"

using namespace rdk::apps;
using namespace rdk::services;
using namespace rdk::examples::apps;

namespace rdk
{
namespace examples
{
namespace apps
{

constexpr const char* EXAMPLE_DESCRIPTION =
    "NVIDIA Rivermax Dev Kit Node Shutdown Video File Sender Example";
constexpr const char* EXAMPLE_USAGE = "Examples:\n"
                                      "  1. node_shutdown_video_file_sender --local-ip 1.2.3.4 --video-file "
                                      "video_1920x1080_60fps_ycbcr422_10bit.raw\n";

/* Two I/O nodes, one stream each, so one can be stopped while the other keeps transmitting. */
constexpr size_t NUM_OF_IO_NODES = 2;
constexpr size_t NUM_OF_STREAMS = NUM_OF_IO_NODES;
/* Time to leave the nodes transmitting before one of them is stopped. A heuristic for a
 * demonstration, not a value to rely on in an application. */
constexpr std::chrono::milliseconds TRANSMIT_TIME{5000};
/* Polling interval. await checks, sleeps out the timeout it is given, and checks again, so the
 * caller chooses the granularity it is polled at. */
constexpr std::chrono::milliseconds AWAIT_SLICE{50};
/* How long to poll for a node, or the application, to finish. */
constexpr std::chrono::milliseconds SHUTDOWN_TIMEOUT{5000};

NodeShutdownVideoFileSenderExample::NodeShutdownVideoFileSenderExample() :
    BaseExample(EXAMPLE_DESCRIPTION, EXAMPLE_USAGE)
{
}

void NodeShutdownVideoFileSenderExample::add_cli_options()
{
    BaseExample::add_cli_options();

    m_cli_parser_manager->add_option(CLIOptStr::VIDEO_FILE);
}

void NodeShutdownVideoFileSenderExample::configure_media_sender_settings(MediaSenderSettings& settings) const
{
    /** Configure network settings */
    settings.local_ips = {m_settings->local_ip};
    settings.destination_ips = {"224.1.1.1"};
    settings.destination_ports = {2000};

    /** Configure video settings */
    settings.video_file = m_settings->video_file;
    settings.dynamic_media_file_load = true;

    settings.media.enable_video = true;
    settings.media.resolution = {1920, 1080};
    settings.media.frame_rate = 60;
    settings.media.sampling_type = VideoSampling::YCbCr_4_2_2;
    settings.media.color_bit_depth = VideoBitDepth::_10;

    /** Configure thread/stream settings: one thread per stream gives one I/O node per stream */
    settings.num_of_threads = NUM_OF_IO_NODES;
    settings.num_of_total_streams = NUM_OF_STREAMS;
    settings.internal_thread_core = 0;
    settings.app_threads_cores = {1, 2};

    /** Configure memory settings */
    settings.allocator_type = AllocatorTypeUI::Auto;
    settings.app_memory_alloc = true;
    settings.header_data_split = true;
}

ReturnStatus NodeShutdownVideoFileSenderExample::operator()()
{
    /** 1. Configure media sender settings with user parameters */
    MediaSenderSettings media_sender_settings;

    media_sender_settings.init_default_values();
    configure_media_sender_settings(media_sender_settings);

    /** 2. Set up media sender settings validation */
    MediaSenderSettingsValidator settings_validator;

    /** 3. Configure user-provided settings builder with validator */
    auto settings_builder = std::make_unique<MediaSenderUserProvidedSettingsBuilder>(
        std::move(media_sender_settings), settings_validator);

    /** 4. Initialize the media sender application */
    MediaSenderApp app(std::move(settings_builder));
    auto status = app.initialize();
    RETURN_FAILURE_ON_ERROR(status, "Failed to initialize Media Sender application");

    /** 5. Read back the I/O nodes the application built, and take the id of the node to stop */
    const auto io_nodes_settings = app.get_io_nodes_settings();
    RETURN_STATUS_IF(io_nodes_settings.size() < NUM_OF_IO_NODES,
        "The application built fewer I/O nodes than configured", ReturnStatus::failure);

    const size_t node_to_stop = io_nodes_settings.front().id;

    /** 6. Start sending media data on another thread */
    ReturnStatus app_status = ReturnStatus::failure;
    std::thread app_thread([&app, &app_status]() { app_status = app.start(); });

    /** 7. Let both nodes transmit, then stop one of them */
    std::this_thread::sleep_for(TRANSMIT_TIME);
    ReturnStatus demo_status = app.stop_io_node(node_to_stop);

    /** 8. Poll until that node has finished */
    std::chrono::milliseconds waited{0};
    do {
        // Success means the node finished; it does not report whether shutdown was clean.
        status = app.await_io_node(node_to_stop, AWAIT_SLICE);
        waited += AWAIT_SLICE;
    } while (status == ReturnStatus::still_active && waited < SHUTDOWN_TIMEOUT);
    if (status != ReturnStatus::success) {
        demo_status = status;
    }

    /** 9. Stop the application, which stops the node it is still running, and wait for it */
    app.stop();
    waited = std::chrono::milliseconds{0};
    do {
        status = app.await(AWAIT_SLICE);
        waited += AWAIT_SLICE;
    } while (status == ReturnStatus::still_active && waited < SHUTDOWN_TIMEOUT);
    if (status != ReturnStatus::success) {
        demo_status = status;
    }

    /** 10. Join the application thread */
    app_thread.join();

    RETURN_FAILURE_ON_ERROR(app_status, "Media Sender application failed");
    RETURN_FAILURE_ON_ERROR(demo_status, "Failed to complete the shutdown demo");

    return ReturnStatus::success;
}

} // namespace apps
} // namespace examples
} // namespace rdk

int main(int argc, const char* argv[])
{
    return rdk::examples::common_example_main<NodeShutdownVideoFileSenderExample>(argc, argv);
}
