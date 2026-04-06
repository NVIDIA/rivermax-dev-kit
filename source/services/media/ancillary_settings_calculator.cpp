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

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "rt_threads.h"

#include "rdk/services/media/ancillary_settings_calculator.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/ulp_packet/writers/rtp_smpte_2110_40_packet_writer.h"
#include "rdk/services/utils/defs.h"

using namespace rdk::services;

ReturnStatus ST_2110_40_MediaSettingsCalculator::calculate_packet_parameters()
{
    // Calculate maximum packet size. Ancillary data packet size will be adjusted at runtime.
    m_media_settings.protocol_header_size = RTP_ST_2110_40_ANCILLARY_HEADER_SIZE;
    m_media_settings.raw_packet_payload_size = MediaSettings::MAX_PAYLOAD_SIZE - m_media_settings.protocol_header_size;
    m_media_settings.packet_payload_size = m_media_settings.protocol_header_size + m_media_settings.raw_packet_payload_size;

    // HDS for ancillary keeps data header with payload
    if (m_media_settings.header_data_split) {
        m_media_settings.packet_app_header_size = m_media_settings.protocol_header_size;
        m_media_settings.packet_payload_size -= m_media_settings.protocol_header_size;
    }

    // Packets per media unit can vary for ancillary streams. Define max amount.
    constexpr uint32_t DEFAULT_MAX_PACKETS_PER_MEDIA_UNIT = 10;
    if (m_media_settings.packets_in_media_unit == 0) {
        m_media_settings.packets_in_media_unit = DEFAULT_MAX_PACKETS_PER_MEDIA_UNIT;
    }
    if (m_media_settings.chunks_in_media_unit == 0) {
        m_media_settings.chunks_in_media_unit = 1;
    }

    // Derive max packets per chunk from other settings. This will be updated at runtime.
    m_media_settings.packets_in_chunk =
        (m_media_settings.packets_in_media_unit + m_media_settings.chunks_in_media_unit - 1) /
        m_media_settings.chunks_in_media_unit;

    return ReturnStatus::success;
}

void ST_2110_40_MediaSettingsCalculator::calculate_timing_parameters()
{
    m_media_settings.sample_rate = MediaSettings::RTP_SAMPLE_RATE;
    m_media_settings.media_unit_time_interval_ns = rational_cast<double>(NS_IN_SEC / m_media_settings.frame_rate);
    m_media_settings.ticks_per_media_unit = m_media_settings.sample_rate / m_media_settings.frame_rate;
    if (m_media_settings.video_scan_type == VideoScanType::Interlaced) {
        m_media_settings.media_unit_time_interval_ns /= 2;
        m_media_settings.ticks_per_media_unit /= 2;
    }
}

void ST_2110_40_MediaSettingsCalculator::calculate_memory_parameters()
{
    if (m_media_settings.media_units_in_mem_block == 0) {
        m_media_settings.media_units_in_mem_block = m_media_settings.DEFAULT_NUM_OF_MEDIA_UNITS_IN_MEM_BLOCK;
    }
    m_media_settings.chunks_in_mem_block = m_media_settings.media_units_in_mem_block * m_media_settings.chunks_in_media_unit;
    m_media_settings.packets_in_mem_block = m_media_settings.chunks_in_mem_block * m_media_settings.packets_in_chunk;
    m_media_settings.bytes_per_media_unit = m_media_settings.raw_packet_payload_size * m_media_settings.packets_in_media_unit;
}

void ST_2110_40_MediaSettingsCalculator::calculate_stride_parameters()
{
    m_media_settings.app_header_stride_size = align_up_pow2(m_media_settings.packet_app_header_size, get_cache_line_size());
    m_media_settings.data_stride_size = align_up_pow2(m_media_settings.packet_payload_size, get_cache_line_size());
}

ReturnStatus ST_2110_40_MediaSettingsCalculator::calculate_media_settings()
{
    // Validate that at least one ancillary data identifier is declared
    if (m_media_settings.data_identifiers.empty()) {
        std::cerr << "Error: No ancillary data identifiers (DID/SDID) declared." << std::endl;
        return ReturnStatus::failure;
    }
    if (m_media_settings.max_ancillary_data_packets_per_packet == 0) {
        std::cerr << "Error: max_ancillary_data_packets_per_packet must be greater than 0." << std::endl;
        return ReturnStatus::failure;
    }
    if (m_media_settings.max_user_data_words_count == 0) {
        std::cerr << "Error: max_user_data_words_count must be greater than 0." << std::endl;
        return ReturnStatus::failure;
    }

    ReturnStatus status = calculate_packet_parameters();
    if (status != ReturnStatus::success) {
        return status;
    }

    calculate_timing_parameters();
    calculate_memory_parameters();
    calculate_stride_parameters();

    return ReturnStatus::success;
}

std::string ST_2110_40_MediaSettingsCalculator::generate_media_sdp(const std::vector<NetworkFlow>& flows)
{
    const auto& first_flow = flows[0];
    const bool is_multi_flow = flows.size() > 1;

    std::vector<std::string> group_mids;
    if (is_multi_flow) {
        for (size_t i = 0; i < flows.size(); ++i) {
            group_mids.push_back(std::string(1, 'a' + static_cast<char>(i)));
        }
    }

    SessionDescription::Builder session_builder(first_flow.source_ip);
    session_builder.set_session_id(SDPManager::generate_ntp_id())
        .set_session_version(SDPManager::generate_ntp_id() + 1)
        .set_session_name("SMPTE ST2110-40");
    if (is_multi_flow) {
        session_builder.add_group(GroupAttribute::Builder("DUP", group_mids).build());
    }

    auto time_description = TimeDescription::Builder().build();
    auto sdp_builder = SDPManager::Builder(session_builder.build(), std::move(time_description));

    for (size_t i = 0; i < flows.size(); ++i) {
        const auto& flow = flows[i];

        SMPTE2110_40_MediaDescription::Builder media_builder {
            flow.destination_port,
            TransportProtocol::RTP_AVP,
            std::to_string(m_media_settings.payload_type),
            flow.destination_ip
        };
        media_builder.set_source_filter(
            SourceFilterAttribute::Builder(flow.destination_ip, flow.source_ip).build()
        );
        for (const auto& identifier : m_media_settings.data_identifiers) {
            media_builder.add_did_sdid(identifier.did, identifier.sdid);
        }
        media_builder.set_extra_format_specific_parameters(m_extra_parameters);

        if (is_multi_flow) {
            media_builder.set_media_id(std::string(1, 'a' + static_cast<char>(i)));
        }

        sdp_builder.add_media_description(media_builder.build());
    }

    return sdp_builder.build()->to_string();
}

std::string ST_2110_40_MediaSettingsCalculator::get_smpte_standard_name() const
{
    return "Ancillary";
}

double ST_2110_40_MediaSettingsCalculator::align_time_to_media_unit_boundary_ns(uint64_t desired_time_ns) const
{
    double media_unit_interval_ns = m_media_settings.media_unit_time_interval_ns;

    // Find the next aligned media unit start time
    uint64_t N = static_cast<uint64_t>(static_cast<double>(desired_time_ns) / media_unit_interval_ns + 1);
    double first_packet_start_time_ns = N * media_unit_interval_ns;

    return first_packet_start_time_ns;
}
