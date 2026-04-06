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

#include "rdk/apps/media_sender/media_sender.h"
#include "rdk/examples/apps/rdk_media_sender/user_provided_settings_video_file_sender.h"

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
    "NVIDIA Rivermax Dev Kit Video File Sender Example Using User-Provided Settings";
constexpr const char* EXAMPLE_USAGE = "Examples:\n"
                                      "  1. user_provided_settings_video_file_sender --local-ip 1.2.3.4 --video-file "
                                      "video_1920x1080_60fps_ycbcr422_10bit.raw\n";

UserProvidedSettingsVideoFileSenderExample::UserProvidedSettingsVideoFileSenderExample() :
    BaseExample(EXAMPLE_DESCRIPTION, EXAMPLE_USAGE)
{
}

void UserProvidedSettingsVideoFileSenderExample::add_cli_options()
{
    BaseExample::add_cli_options();

    m_cli_parser_manager->add_option(CLIOptStr::VIDEO_FILE);
}

void UserProvidedSettingsVideoFileSenderExample::configure_media_sender_settings(MediaSenderSettings& settings) const
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

    /** Configure thread/stream settings */
    settings.num_of_threads = 1;
    settings.num_of_total_streams = 1;
    settings.internal_thread_core = 0;
    settings.app_threads_cores = {1};

    /** Configure memory settings */
    settings.allocator_type = AllocatorTypeUI::Auto;
    settings.app_memory_alloc = true;
    settings.header_data_split = true;
}

ReturnStatus UserProvidedSettingsVideoFileSenderExample::operator()()
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

    /** 5. Start sending media data */
    status = app.run();
    RETURN_FAILURE_ON_ERROR(status, "Failed to run Media Sender application");

    return ReturnStatus::success;
}

} // namespace apps
} // namespace examples
} // namespace rdk

int main(int argc, const char* argv[])
{
    return rdk::examples::common_example_main<UserProvidedSettingsVideoFileSenderExample>(argc, argv);
}
