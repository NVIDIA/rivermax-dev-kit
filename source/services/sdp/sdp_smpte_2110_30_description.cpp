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
#include <sstream>
#include <iomanip>

#include "rdk/services/utils/enum_utils.h"
#include "rdk/services/sdp/sdp_defs.h"
#include "rdk/services/sdp/sdp_common_descriptions.h"
#include "rdk/services/sdp/sdp_smpte_2110_30_description.h"

using namespace rivermax::dev_kit::services;

std::vector<json> SMPTE2110_30_MediaDescription::get_media_description_attributes() const
{
    std::vector<FormatSpecificParameter> format_specific_parameters = {};
    if (!m_channel_order_values.empty()) {
        format_specific_parameters.push_back({"channel-order", std::string("SMPTE2110.(") + m_channel_order_values + ")"});
    }

    format_specific_parameters.insert(
        format_specific_parameters.end(),
        m_extra_format_specific_parameters.begin(),
        m_extra_format_specific_parameters.end()
    );

    std::vector<json> attributes = {
        get_rtp_map_attribute({RTPMapAttribute{
            m_payload_type,
            enum_to_string(m_encoding),
            static_cast<size_t>(std::stoul(enum_to_string(m_sampling_rate))),
            std::to_string(m_channels)
        }})
    };

    attributes.push_back(get_media_format_specific_attribute({MediaFormatAttribute{m_media_format,
                                                              std::move(format_specific_parameters)}}));

    attributes.push_back(get_ptime_attribute(m_ptime_ms));
    attributes.push_back(get_maxptime_attribute(m_maxptime_ms));
    attributes.push_back(get_media_clock_attribute(m_media_clock));
    attributes.push_back(get_ref_clock_timestamp_attribute(
        m_timestamp_ref_clock,
        m_timestamp_ref_clock_ptp_grandmaster_clock_identity,
        m_timestamp_ref_clock_ptp_domain_number,
        m_timestamp_ref_clock_ptp_traceable,
        m_timestamp_ref_clock_local_mac
    ));

    if (m_source_filter) {
        attributes.insert(attributes.begin(), get_source_filter_attribute(*m_source_filter));
    }

    if (!m_media_id.empty()) {
        attributes.push_back(get_media_id_attribute(m_media_id));
    }

    return attributes;
}

SMPTE2110_30_MediaDescription::Builder&
SMPTE2110_30_MediaDescription::Builder::set_channel_order(std::initializer_list<ChannelGroupingSymbol> symbols)
{
    if (symbols.size() == 0) {
        throw "Invalid channel-order: empty list";
    }

    auto to_short_code = [](ChannelGroupingSymbol s) -> std::string {
        switch (s) {
            case ChannelGroupingSymbol::M: return "M";
            case ChannelGroupingSymbol::DM: return "DM";
            case ChannelGroupingSymbol::ST: return "ST";
            case ChannelGroupingSymbol::LtRt: return "LtRt";
            case ChannelGroupingSymbol::_51: return "51";
            case ChannelGroupingSymbol::_71: return "71";
            case ChannelGroupingSymbol::_222: return "222";
            case ChannelGroupingSymbol::SGRP: return "SGRP";
            default:
                break;
        }

        int v = static_cast<int>(s);
        if (v >= static_cast<int>(ChannelGroupingSymbol::U01) && v <= static_cast<int>(ChannelGroupingSymbol::U64)) {
            std::ostringstream oss;
            oss << 'U' << std::setw(2) << std::setfill('0') << v - static_cast<int>(ChannelGroupingSymbol::U01) + 1;
            return oss.str();
        }
        return std::string();
    };

    std::string values;
    bool first = true;
    for (ChannelGroupingSymbol s : symbols) {
        std::string code = to_short_code(s);
        if (code.empty()) {
            throw "Invalid channel-order token";
        }
        if (!first) {
            values.push_back(',');
        }
        values.append(code);
        first = false;
    }
    return set(m_instance->m_channel_order_values, values);
}
