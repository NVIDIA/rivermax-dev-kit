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

#include <iostream>
#include <string>

#include "rdk/services/sdp/sdp_manager.h"
#include "rdk/services/sdp/sdp_smpte_2110_40_description.h"
#include "rdk/examples/services/sdp/sdp_smpte_2110_40_description.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::examples::services;

namespace rivermax
{
namespace dev_kit
{
namespace examples
{
namespace services
{

constexpr const char* EXAMPLE_DESCRIPTION = "NVIDIA Rivermax Dev Kit SMPTE 2110-40 SDP String Generation Example";
constexpr const char* EXAMPLE_USAGE = "Examples:\n"
                                      "  1. generate_smpte_2110_40_sdp";

SMPTE_2110_40_SDPGenerationExample::SMPTE_2110_40_SDPGenerationExample() :
    BaseExample(EXAMPLE_DESCRIPTION, EXAMPLE_USAGE)
{
}

ReturnStatus SMPTE_2110_40_SDPGenerationExample::operator()()
{
    /** Example network parameters */
    constexpr auto local_ip = "1.1.1.1";
    constexpr auto destination_ip = "224.1.2.3";
    constexpr auto destination_port = 5004;

    /** 1. Create session description */
    auto session_description = SessionDescription::Builder(local_ip)
        .set_session_name("SMPTE ST2110-40")
        .build();

    /** 2. Create time description */
    auto time_description = TimeDescription::Builder()
        .set_start_time(SDPManager::generate_ntp_id())
        .set_stop_time(SDPManager::generate_ntp_id() + 1)
        .build();

    /** 3. Create media description */
    auto media_description = SMPTE2110_40_MediaDescription::Builder(
        destination_port, TransportProtocol::RTP_AVP, "100", destination_ip)
        .set_source_filter(SourceFilterAttribute::Builder(destination_ip, local_ip).build())
        .set_did_sdid(0x61, 0x01)
        .set_vpid_code(0x84)
        .build();

    /** 4. Create SDP string */
    auto sdp_string = SDPManager::Builder(std::move(session_description), std::move(time_description))
        .add_media_description(std::move(media_description))
        .build()->to_string();

    std::cout << "Generated SDP string:\n" << sdp_string << std::endl;

    return ReturnStatus::success;
}

} // namespace services
} // namespace examples
} // namespace dev_kit
} // namespace rivermax

int main(int argc, const char* argv[])
{
    return rivermax::dev_kit::examples::common_example_main<SMPTE_2110_40_SDPGenerationExample>(argc, argv);
}
