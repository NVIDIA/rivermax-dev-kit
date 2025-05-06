/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "rdk/apps/rmax_base_app.h"
#include "rdk/apps/rmax_ipmx_sender/rmax_ipmx_sender.h"

using namespace rivermax::dev_kit::apps;
using namespace rivermax::dev_kit::apps::rmax_ipmx_sender;

/**
 * Application constants.
 */
constexpr const char* APP_DESCRIPTION = "NVIDIA Rivermax IPMX sender demo app ";

constexpr const char* APP_EXAMPLES = \
    "\nExamples:\n"
    "  1. rmax_ipmx_sender --local-ip 1.2.3.4 --video-resolution 1920x1080 "
    "--video-frame-rate 60 -v --ptp\n"
    "  2. rmax_ipmx_sender --local-ip 1.2.3.4 --video-resolution 1920x1080 "
    "--video-frame-rate 60 --dst-ip 234.5.6.7 --dst-port 2000\n"
    "  3. rmax_ipmx_sender --local-ip 1.2.3.4 --video-resolution 1920x1080 "
    "--video-frame-rate 60 --streams 10\n"
    "  4. rmax_ipmx_sender --local-ip 1.2.3.4 --video-resolution 1920x1080 "
    "--video-frame-rate 60 --threads 2 --streams 10 -a 1,2 -c 3\n";

int main(int argc, const char* argv[])
{
    return common_cli_main<IPMXSenderApp, IPMXSenderCLISettingsBuilder, IPMXSenderSettingsValidator>(
        argc, argv, APP_DESCRIPTION, APP_EXAMPLES);
}
