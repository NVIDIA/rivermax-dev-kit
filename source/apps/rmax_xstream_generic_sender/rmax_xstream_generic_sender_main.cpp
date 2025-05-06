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
#include "rdk/apps/rmax_xstream_generic_sender/rmax_xstream_generic_sender.h"

using namespace rivermax::dev_kit::apps;
using namespace rivermax::dev_kit::apps::rmax_xstream_generic_sender;

/**
 * Application constants.
 */
constexpr const char* APP_DESCRIPTION = "NVIDIA Rivermax generic sender demo app ";
constexpr const char* APP_EXAMPLES = \
    "\nExamples:\n"
    "  1. rmax_xstream_generic_sender --local-ip 1.2.3.4\n"
    "  2. rmax_xstream_generic_sender --local-ip 1.2.3.4 --dst-ip 1.2.5.6 --dst-port 56789\n"
    "  3. rmax_xstream_generic_sender --local-ip 1.2.3.4 --threads 10 --streams 20 --flows "
    "100 --internal-core 2 --application-core 2,4,6\n"
    "  4. rmax_xstream_generic_sender --local-ip 1.2.3.4 --streams 2 --flows 1000000 --chunks 2 "
    "--packets 4096 --payload-size 64\n"
    "  5. rmax_xstream_generic_sender --local-ip 1.2.3.4 --streams 2 --flows 10 --packets 4 "
    "--payload-size 512 --rate-bps 5000000 --rate-burst 1 -v\n";

int main(int argc, const char* argv[])
{
    return common_cli_main<GenericSenderApp, GenericSenderCLISettingsBuilder, GenericSenderSettingsValidator>(
        argc, argv, APP_DESCRIPTION, APP_EXAMPLES);
}
