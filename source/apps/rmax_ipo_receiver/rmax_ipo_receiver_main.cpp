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
#include "rdk/apps/rmax_ipo_receiver/rmax_ipo_receiver.h"

using namespace rivermax::dev_kit::apps;
using namespace rivermax::dev_kit::apps::rmax_ipo_receiver;

/**
 * Application constants.
 */
constexpr const char* APP_DESCRIPTION = "NVIDIA Rivermax IPO receiver demo app ";
constexpr const char* APP_EXAMPLES = \
    "\nExamples:\n"
    "  1. rmax_ipo_receiver --local-ips 1.2.3.4 --src-ips 6.7.8.9 --dst-ips 1.2.3.4 -p 50020 -v\n"
    "  2. rmax_ipo_receiver --local-ips 1.2.3.4,1.2.3.5 --src-ips 6.7.8.9,6.7.8.10 --dst-ips 1.2.3.4,1.2.3.5 -p 50020,50120 -v\n"
    "  3. rmax_ipo_receiver --local-ips 1.2.3.4,1.2.3.5 --src-ips 6.7.8.9,6.7.8.10 --dst-ips 1.2.3.4,1.2.3.5 -p 50020,50120 --app-hdr-size 50 -v\n"
    "  4. rmax_ipo_receiver --local-ips 1.2.3.4,1.2.3.5 --src-ips 6.7.8.9,6.7.8.10 --dst-ips 239.1.1.1,239.1.1.2 -p 50020,50120 -v\n"
    "  5. rmax_ipo_receiver --local-ips 1.2.3.4,1.2.3.5 --src-ips 6.7.8.9,6.7.8.10 --dst-ips 239.1.1.1,239.1.1.2 -p 50020,50120 --threads 2 --streams 10 -a 1,2 -i 3\n";

int main(int argc, const char* argv[])
{
    return common_cli_main<IPOReceiverApp, IPOReceiverCLISettingsBuilder, IPOReceiverSettingsValidator>(
        argc, argv, APP_DESCRIPTION, APP_EXAMPLES);
}
