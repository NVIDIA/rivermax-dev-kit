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

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

#include "rdk/services/utils/enum_utils.h"
#include "rdk/services/sdp/sdp_defs.h"
#include "rdk/services/sdp/sdp_common_descriptions.h"
#include "rdk/services/sdp/sdp_smpte_2110_40_description.h"

using namespace rdk::services;

std::vector<json> SMPTE2110_40_MediaDescription::get_media_description_attributes() const
{
    std::vector<FormatSpecificParameter> format_specific_parameters = {};

    for (const auto& pair : m_did_sdids) {
        std::ostringstream oss;
        oss << '{'
            << "0x" << std::uppercase << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(pair.did)
            << ','
            << "0x" << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(pair.sdid)
            << '}';
        format_specific_parameters.push_back({"DID_SDID", oss.str(), true});
    }
    if (m_vpid_code_set) {
        format_specific_parameters.push_back({"VPID_Code", std::to_string(static_cast<unsigned int>(m_vpid_code)), true});
    }

    format_specific_parameters.insert(
        format_specific_parameters.end(),
        m_extra_format_specific_parameters.begin(),
        m_extra_format_specific_parameters.end()
    );

    std::vector<json> attributes = {
        get_rtp_map_attribute({RTPMapAttribute{
            m_payload_type,
            "smpte291",
            ANCILLARY_RTP_CLOCK_RATE,
            ""
        }}),
        get_media_format_specific_attribute({MediaFormatAttribute{m_media_format, std::move(format_specific_parameters)}}),
        get_media_clock_attribute(m_media_clock),
        get_ref_clock_timestamp_attribute(
            m_timestamp_ref_clock,
            m_timestamp_ref_clock_ptp_grandmaster_clock_identity,
            m_timestamp_ref_clock_ptp_domain_number,
            m_timestamp_ref_clock_ptp_traceable,
            m_timestamp_ref_clock_local_mac
        )
    };

    if (m_source_filter) {
        attributes.insert(attributes.begin(), get_source_filter_attribute(*m_source_filter));
    }

    if (!m_media_id.empty()) {
        attributes.push_back(get_media_id_attribute(m_media_id));
    }

    return attributes;
}
