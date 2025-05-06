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

#include <cstddef>
#include <string>

#include "sdptransform.hpp"

#include "rdk/services/utils/enum_utils.h"
#include "rdk/services/sdp/sdp_common_descriptions.h"

using namespace rivermax::dev_kit::services;

/**
 * @brief: Converts the given network type to a string.
 *
 * @param [in] network_type: The network type.
 *
 * @return: The string representation of the network type.
 */
static inline std::string ip_version_to_string(AddressType address_type)
{
    return address_type == AddressType::IP4
               ? "IP4"
               : (address_type == AddressType::IP6 ? "IP6" : enum_to_string(AddressType::Unknown));
}

SessionDescription::operator json() const
{
    return {
        {"version", m_protocol_version},
        {"origin",
         {{"username", m_username},
          {"sessionId", m_session_id},
          {"sessionVersion", m_session_version},
          {"netType", enum_to_string(m_network_type)},
          {"ipVer", enum_to_string(m_address_type)},
          {"address", m_unicast_address}}},
        {"name", m_session_name}
    };
}

TimeDescription::operator json() const
{
    return {{"timing", {{"start", m_start_time}, {"stop", m_stop_time}}}};
}

SourceFilterAttribute::operator json() const
{
    return {
        {"sourceFilter",
         {{"filterMode", enum_to_string(filter_mode)},
          {"netType", enum_to_string(network_type)},
          {"addressTypes", ip_version_to_string(address_type)},
          {"destAddress", destination_address},
          {"srcList", source_list}}}
    };
}

BaseMediaDescription::operator json() const
{
    json media_description = {
        {"type", enum_to_string(m_media_type)},
        {"port", m_transport_port},
        {"protocol", enum_to_string(m_transport_protocol)},
        {"payloads", m_media_format_description},
        {"connection",
         {{"version", enum_to_string(m_address_type)}, {"ip", m_connection_address}, {"ttl", m_connection_ttl}}}
    };

    auto attributes = get_media_description_attributes();
    for (const auto& attribute : attributes) {
        media_description.merge_patch(attribute);
    }

    return media_description;
}

json BaseMediaDescription::get_ref_clock_timestamp_attribute(
    TimestampRefClock timestamp_ref_clock,
    const std::string& timestamp_ref_clock_ptp_grandmaster_clock_identity,
    size_t timestamp_ref_clock_ptp_domain_number,
    bool timestamp_ref_clock_ptp_traceable,
    const std::string& timestamp_ref_clock_local_mac) const
{
    std::string value;
    if (timestamp_ref_clock == TimestampRefClock::PTP) {
        if (timestamp_ref_clock_ptp_traceable) {
            value = "ptp=IEEE1588-2008:traceable";
        } else {
            value = "ptp=IEEE1588-2008:" + timestamp_ref_clock_ptp_grandmaster_clock_identity + ":" +
                    std::to_string(timestamp_ref_clock_ptp_domain_number);
        }
    } else if (timestamp_ref_clock == TimestampRefClock::LocalMAC) {
        value = "localmac=" + timestamp_ref_clock_local_mac;
    } else {
        value = enum_to_string(timestamp_ref_clock);
    }
    return {{"tsRefclk", value}};
}

json BaseMediaDescription::get_media_clock_attribute(MediaClock media_clock) const
{
    return {
        {"mediaclk",
         media_clock == MediaClock::Direct ? enum_to_string(media_clock) + "=0" : enum_to_string(media_clock)}
    };
}

json BaseMediaDescription::get_media_format_specific_attribute(const std::vector<MediaFormatAttribute>& formats) const
{
    json fmtp_list = json::array();

    for (const auto& format : formats) {
        std::string format_specific_str;
        for (const auto& parameter : format.parameters) {
            format_specific_str += parameter.name + (parameter.value.empty() ? "" : "=" + parameter.value) + "; ";
        }
        if (!format_specific_str.empty()) {
            format_specific_str.erase(format_specific_str.size() - 2);
        }
        fmtp_list.push_back({{"payload", format.format}, {"config", std::move(format_specific_str)}});
    }

    return {{"fmtp", std::move(fmtp_list)}};
}

json BaseMediaDescription::get_rtp_map_attribute(const std::vector<RTPMapAttribute>& formats) const
{
    json rtpmap_list = json::array();
    for (const auto& format : formats) {
        rtpmap_list.push_back(
            {{"payload", format.payload_type},
             {"codec", format.encoding_name},
             {"rate", format.clock_rate},
             {"encoding", format.encoding_parameters}}
        );
    }

    return {{"rtp", rtpmap_list}};
}
