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

#include "rdk/apps/base_app.h"
#include "rdk/apps/media_sender/media_sender.h"

using namespace rivermax::dev_kit::apps;
using namespace rivermax::dev_kit::apps::media_sender;

/**
 * Application constants.
 */
constexpr const char* APP_DESCRIPTION = "NVIDIA Rivermax Dev Kit media sender demo app";
constexpr const char* APP_EXAMPLES = \
    "\nExamples:\n"
    "  1. rdk_media_sender --local-ips 1.2.3.4 --enable-video --video-resolution"
    " 1920x1080 --video-frame-rate 60 -v\n"
    "  2. rdk_media_sender --local-ips 1.2.3.4 --enable-video --video-resolution"
    " 1920x1080 --video-frame-rate 60 --dst-ips 234.5.6.7 --dst-ports 2000\n"
    "  3. rdk_media_sender --local-ips 1.2.3.4 --enable-video --video-resolution"
    " 1920x1080 --video-frame-rate 60 --streams 10\n"
    "  4. rdk_media_sender --local-ips 1.2.3.4 --enable-video --video-resolution"
    " 1920x1080 --video-frame-rate 60 --threads 2 --streams 10 -a 1,2 -c 3\n"
    "  5. rdk_media_sender --local-ips 1.2.3.4 --enable-video --video-resolution"
    " 1920x1080 --video-frame-rate 30 --video-scan-type interlaced -v\n"
    "  6. rdk_media_sender --local-ips 1.2.3.4 --dst-ips 224.1.1.2 --dst-ports 50001 --enable-video --vr"
    " 1920x1080 --vfr 60 --vs 1 -m -M -A 2 -r -g 0 -v -i 5 -a 9 --vf input_file.ycbcr --dfl\n"
    "  7. rdk_media_sender --local-ips 1.2.3.4 -m --enable-audio --audio-file"
    " input_file.wav --audio-sampling-rate 48000 --audio-encoding L24 --audio-ptime-us 1000\n"
    "  8. rdk_media_sender --local-ips 1.2.3.4 --enable-ancillary\n"
    "  9. rdk_media_sender --local-ips 1.2.3.4 -m --enable-video --vf video_file.ycbcr"
    " --enable-audio --af audio_file.wav --asr 48000 --enable-ancillary\n"
    "  10. rdk_media_sender --local-ips 1.2.3.4,1.2.3.5 --dst-ips 224.1.1.2,224.1.1.3"
    " --dst-ports 5000,5000 --enable-video --enable-redundancy -v\n";

int main(int argc, const char* argv[])
{
    return common_cli_main<MediaSenderApp, MediaSenderCLISettingsBuilder, MediaSenderSettingsValidator>(
        argc, argv, APP_DESCRIPTION, APP_EXAMPLES);
}
