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
#include "rdk/examples/apps/media_sender/cli_settings_media_file_sender.h"

using namespace rdk::apps;
using namespace rdk::services;
using namespace rdk::examples::apps;

namespace rdk
{
namespace examples
{
namespace apps
{

constexpr const char* EXAMPLE_DESCRIPTION = "NVIDIA Rivermax Dev Kit Media File Sender Example Using CLI Settings";
constexpr const char* EXAMPLE_USAGE =
    "Examples:\n"
    "  1. cli_settings_media_file_sender --local-ip 1.2.3.4 -m --enable-video --video-file video.raw\n"
    "  2. cli_settings_media_file_sender --local-ip 1.2.3.4 --dst-ip 224.1.1.1 --dst-port 50001 -m "
    "--enable-video --video-file video.raw --dynamic-file-loading\n"
    "  3. cli_settings_media_file_sender --local-ip 1.2.3.4 -m --enable-video --video-file video.raw "
    "--video-frame-rate 30 --video-resolution 1920x1080\n"
    "  4. cli_settings_media_file_sender --local-ip 1.2.3.4 -m --enable-video --video-file video.raw "
    "--threads 2 --streams 4\n"
    "  5. cli_settings_media_file_sender --local-ip 1.2.3.4 -m--enable-audio --audio-file audio.raw "
    "--audio-ptime-us 250 --audio-sampling-rate 48000 --audio-encoding L24\n"
    "  6. cli_settings_media_file_sender --local-ip 1.2.3.4 -m --enable-audio --audio-file audio.raw "
    "--dynamic-file-loading --audio-ptime-us 1000 --audio-sampling-rate 96000 --audio-encoding L16\n";

CLISettingsMediaFileSenderExample::CLISettingsMediaFileSenderExample() :
    BaseExample(EXAMPLE_DESCRIPTION, EXAMPLE_USAGE)
{
}

ReturnStatus CLISettingsMediaFileSenderExample::operator()()
{
    /** 1. Set up media sender settings validation */
    MediaSenderSettingsValidator settings_validator;

    /** 2. Configure CLI settings builder with validator */
    auto settings_builder = std::make_unique<MediaSenderCLISettingsBuilder>(
        m_argc, m_argv, EXAMPLE_DESCRIPTION, EXAMPLE_USAGE, settings_validator);

    /** 3. Initialize the media sender application */
    MediaSenderApp app(std::move(settings_builder));
    auto status = app.initialize();
    if (status == ReturnStatus::success_cli_help) {
        return ReturnStatus::success;
    }
    RETURN_FAILURE_ON_ERROR(status, "Failed to initialize Media Sender application");

    /** 4. Start sending media data */
    status = app.run();
    RETURN_FAILURE_ON_ERROR(status, "Failed to run Media Sender application");

    return ReturnStatus::success;
}

int CLISettingsMediaFileSenderExample::run(int argc, const char* argv[])
{
    try {
        m_argc = argc;
        m_argv = argv;

        auto status = this->operator()();
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to run example with status: " << static_cast<int>(status) << std::endl;
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Failed to run example with exception: " << error.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Failed to run example with unknown exception" << std::endl;
        return EXIT_FAILURE;
    }
}

} // namespace apps
} // namespace examples
} // namespace rdk

int main(int argc, const char* argv[])
{
    return rdk::examples::common_example_main<CLISettingsMediaFileSenderExample>(argc, argv);
}
