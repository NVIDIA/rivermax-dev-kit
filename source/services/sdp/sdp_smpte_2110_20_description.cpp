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

#include <string>
#include <vector>

#include "sdptransform.hpp"

#include "rdk/services/utils/enum_utils.h"
#include "rdk/services/sdp/sdp_defs.h"
#include "rdk/services/sdp/sdp_common_descriptions.h"
#include "rdk/services/sdp/sdp_smpte_2110_20_description.h"

using namespace rivermax::dev_kit::services;

std::vector<json> SMPTE2110_20_MediaDescription::get_media_description_attributes() const
{
    std::vector<FormatSpecificParameter> format_specific_parameters = {
        {"sampling", enum_to_string(m_sampling)},
        {"width", std::to_string(m_width)},
        {"height", std::to_string(m_height)},
        {"exactframerate", m_exact_frame_rate},
        {"depth", enum_to_string(m_depth)},
        {"colorimetry", enum_to_string(m_colorimetry)},
        {"PM", enum_to_string(m_packaging_mode)},
        {"SSN", enum_to_string(m_smpte_standard_number)},
        {"TP", enum_to_string(m_sender_type)},
    };

    std::vector<FormatSpecificParameter> format_specific_conditional_parameters = {
        {"interlace", "", m_video_scan_type == VideoScanType::Interlaced},
        {"CMAX", std::to_string(m_cmax), m_cmax > 0},
        {"MAXUDP", std::to_string(m_max_udp), m_max_udp > 0},
        {"TSMODE", enum_to_string(m_timestamp_mode), m_timestamp_mode < TimestampMode::Unknown},
        {"TSDELAY", std::to_string(m_timestamp_delay), m_timestamp_delay > 0}
    };
    format_specific_conditional_parameters.insert(
        format_specific_conditional_parameters.end(),
        m_extra_format_specific_parameters.begin(),
        m_extra_format_specific_parameters.end()
    );

    for (const auto& parameter : format_specific_conditional_parameters) {
        if (parameter.condition) {
            format_specific_parameters.push_back({parameter.name, parameter.value});
        }
    }

    std::vector<json> attributes = {
        get_rtp_map_attribute({RTPMapAttribute{m_payload_type, "raw", 90000, ""}}),
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

    return attributes;
}
