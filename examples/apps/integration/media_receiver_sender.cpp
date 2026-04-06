/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <thread>

#include "rdk/apps/rtp_receiver/rtp_receiver.h"
#include "rdk/apps/media_sender/media_sender.h"
#include "rdk/examples/apps/integration/media_receiver_sender.h"

using namespace rdk::apps;
using namespace rdk::services;
using namespace rdk::examples::apps::integration;

namespace rdk
{
namespace examples
{
namespace apps
{
namespace integration
{

constexpr const char* EXAMPLE_DESCRIPTION = "NVIDIA Rivermax Dev Kit Combined Receiver and Sender Example";
constexpr const char* EXAMPLE_USAGE = "Examples:\n"
                                      "  1. rmax_media_receiver_sender --local-ip 1.2.3.4 --source-ip 5.6.7.8";

ReceiverSenderExample::ReceiverSenderExample() :
    BaseExample(EXAMPLE_DESCRIPTION, EXAMPLE_USAGE)
{
}

void ReceiverSenderExample::add_cli_options()
{
    BaseExample::add_cli_options();

    m_cli_parser_manager->add_option(CLIOptStr::SRC_IP);
}

void ReceiverSenderExample::configure_rtp_receiver_media_sender_settings(
    RTPReceiverSettings& rtp_receiver_settings,
    MediaSenderSettings& media_sender_settings) const
{
    /** Configure network settings */
    rtp_receiver_settings.source_ip = m_settings->source_ip;
    rtp_receiver_settings.local_ip = m_settings->local_ip;
    rtp_receiver_settings.destination_ip = "224.1.1.1";
    rtp_receiver_settings.destination_port = 2000;
    media_sender_settings.local_ips = {m_settings->local_ip};
    media_sender_settings.destination_ips = {"224.2.2.2"};
    media_sender_settings.destination_ports = {3000};

    /** Configure RTP receiver settings */
    rtp_receiver_settings.is_extended_sequence_number = false;
    rtp_receiver_settings.num_of_packets_in_chunk = RTPReceiverSettings::DEFAULT_NUM_OF_PACKETS_IN_CHUNK;

    /** Configure video sender settings */
    media_sender_settings.media.enable_video = true;
    media_sender_settings.media.resolution = {1920, 1080};
    media_sender_settings.media.frame_rate = 60;
    media_sender_settings.media.sampling_type = VideoSampling::YCbCr_4_2_2;
    media_sender_settings.media.color_bit_depth = VideoBitDepth::_10;

    /** Configure thread/stream settings */
    rtp_receiver_settings.num_of_threads = media_sender_settings.num_of_threads = 1;
    rtp_receiver_settings.num_of_total_streams = media_sender_settings.num_of_total_streams = 1;
    rtp_receiver_settings.internal_thread_core = media_sender_settings.internal_thread_core = 0;
    rtp_receiver_settings.app_threads_cores = {1};
    media_sender_settings.app_threads_cores = {2};

    /** Configure memory settings */
    rtp_receiver_settings.allocator_type = media_sender_settings.allocator_type = AllocatorTypeUI::Auto;
    rtp_receiver_settings.app_memory_alloc = media_sender_settings.app_memory_alloc = true;
    rtp_receiver_settings.header_data_split = media_sender_settings.header_data_split = true;
}

ReturnStatus ReceiverSenderExample::operator()()
{
    /** 1. Configure RTP receiver and media sender settings with user parameters */
    RTPReceiverSettings rtp_receiver_settings;
    MediaSenderSettings media_sender_settings;

    rtp_receiver_settings.init_default_values();
    media_sender_settings.init_default_values();
    configure_rtp_receiver_media_sender_settings(rtp_receiver_settings, media_sender_settings);

    /** 2. Set up RTP receiver and media sender settings validation */
    RTPReceiverSettingsValidator rtp_receiver_settings_validator;
    MediaSenderSettingsValidator media_sender_settings_validator;

    /** 3. Configure user-provided settings builders with validators */
    auto rtp_receiver_settings_builder = std::make_unique<RTPReceiverUserProvidedSettingsBuilder>(
        std::move(rtp_receiver_settings), rtp_receiver_settings_validator);
    auto media_sender_settings_builder = std::make_unique<MediaSenderUserProvidedSettingsBuilder>(
        std::move(media_sender_settings), media_sender_settings_validator);

    /** 4. Initialize the RTP Receiver and Media Sender applications */
    RTPReceiverApp rtp_receiver_app(std::move(rtp_receiver_settings_builder));
    MediaSenderApp media_sender_app(std::move(media_sender_settings_builder));

    auto status = rtp_receiver_app.initialize();
    RETURN_FAILURE_ON_ERROR(status, "Failed to initialize RTP Receiver application");
    status = media_sender_app.initialize();
    RETURN_FAILURE_ON_ERROR(status, "Failed to initialize Media Sender application");

    /** 5. Start receiving RTP chunks and sending video frames in separate threads */
    std::thread receiver_thread(&RTPReceiverApp::run, &rtp_receiver_app);
    std::thread sender_thread(&MediaSenderApp::run, &media_sender_app);

    receiver_thread.join();
    sender_thread.join();

    return ReturnStatus::success;
}

} // namespace integration
} // namespace apps
} // namespace examples
} // namespace rdk

int main(int argc, const char* argv[])
{
    return rdk::examples::common_example_main<ReceiverSenderExample>(argc, argv);
}
