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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "rt_threads.h"

#include "rdk/services/media/audio_settings_calculator.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/media/media_defs.h"
#include "rdk/services/utils/defs.h"
#include "rdk/services/utils/enum_utils.h"

using namespace rivermax::dev_kit::services;

// Audio sampling rate to Hz value mapping
const std::unordered_map<AudioSamplingRate, uint32_t> AUDIO_SAMPLING_RATE_MAP = {
    {AudioSamplingRate::_44100, 44100},
    {AudioSamplingRate::_48000, 48000},
    {AudioSamplingRate::_96000, 96000}
};

// Audio encoding to bit depth mapping
const std::unordered_map<AudioEncoding, uint32_t> AUDIO_ENCODING_BIT_DEPTH_MAP = {
    {AudioEncoding::L16, 16},
    {AudioEncoding::L24, 24}
};

bool ST_2110_30_MediaSettingsCalculator::is_channel_count_supported(uint8_t num_channels)
{
    return std::find(SUPPORTED_AUDIO_CHANNEL_COUNTS.begin(), SUPPORTED_AUDIO_CHANNEL_COUNTS.end(), num_channels)
           != SUPPORTED_AUDIO_CHANNEL_COUNTS.end();
}

ReturnStatus ST_2110_30_MediaSettingsCalculator::calculate_packet_parameters()
{
    m_media_settings.packets_per_second = USEC_IN_SEC / m_media_settings.ptime_usec;

    auto rate_it = AUDIO_SAMPLING_RATE_MAP.find(m_media_settings.sampling_rate);
    if (rate_it == AUDIO_SAMPLING_RATE_MAP.end()) {
        std::cerr << "Error: Unsupported audio sampling rate: "
                  << static_cast<int>(m_media_settings.sampling_rate) << std::endl;
        return ReturnStatus::failure;
    }
    uint32_t sampling_rate_value = rate_it->second;
    m_media_settings.samples_per_packet = sampling_rate_value / m_media_settings.packets_per_second;

    auto encoding_it = AUDIO_ENCODING_BIT_DEPTH_MAP.find(m_media_settings.encoding);
    if (encoding_it == AUDIO_ENCODING_BIT_DEPTH_MAP.end()) {
        std::cerr << "Error: Unsupported audio encoding: "
                  << static_cast<int>(m_media_settings.encoding) << std::endl;
        return ReturnStatus::failure;
    }
    m_media_settings.bytes_per_sample = encoding_it->second / 8;

    // Calculate payload size (samples * channels * bytes per sample)
    size_t payload_size = m_media_settings.samples_per_packet *
                          m_media_settings.num_channels *
                          m_media_settings.bytes_per_sample;

    m_media_settings.protocol_header_size = RTP_HEADER_SIZE;
    m_media_settings.raw_packet_payload_size = static_cast<uint16_t>(payload_size);
    m_media_settings.packet_payload_size = static_cast<uint16_t>(payload_size + m_media_settings.protocol_header_size);

    // Validate packet size doesn't exceed UDP limit
    if (m_media_settings.packet_payload_size > MediaSettings::MAX_PAYLOAD_SIZE) {
        std::cerr << "Error: Audio packet size (" << m_media_settings.packet_payload_size
                  << " bytes) exceeds network limit (" << MediaSettings::MAX_PAYLOAD_SIZE << " bytes). "
                  << "Consider shorter ptime or fewer channels/lower bit depth." << std::endl;
        return ReturnStatus::failure;
    }

    if (m_media_settings.header_data_split) {
        m_media_settings.packet_app_header_size = m_media_settings.protocol_header_size;
        m_media_settings.packet_payload_size -= m_media_settings.protocol_header_size;
    }

    // Set default packets per media unit if not provided
    if (!m_media_settings.packets_in_media_unit) {
        constexpr uint32_t PACKETS_PER_MEDIA_UNIT = 100;
        m_media_settings.packets_in_media_unit = PACKETS_PER_MEDIA_UNIT;
    }

    // Validate and apply custom chunk size if provided
    bool chunk_size_applied = false;
    if (m_media_settings.packets_in_chunk > 0) {
        if (m_media_settings.packets_in_media_unit % m_media_settings.packets_in_chunk == 0) {
            chunk_size_applied = true;
            std::cout << "Using custom audio chunk size: " << m_media_settings.packets_in_chunk
                      << " packets per chunk" << std::endl;
        } else {
            std::cerr << "Warning: Custom chunk size (" << m_media_settings.packets_in_chunk
                      << ") is not a divisor of packets in media unit ("
                      << m_media_settings.packets_in_media_unit << "). Calculating optimal size." << std::endl;
        }
    }

    if (!chunk_size_applied) {
        // Aim for ~20 packets per chunk (20ms chunks if ptime=1ms)
        constexpr uint32_t TARGET_PACKETS_PER_CHUNK = 20;

        m_media_settings.packets_in_chunk = m_media_settings.packets_in_media_unit;
        for (uint32_t chunk_cnt = 1; chunk_cnt <= m_media_settings.packets_in_media_unit; chunk_cnt++) {
            if (m_media_settings.packets_in_media_unit % chunk_cnt != 0) {
                continue;
            }
            uint32_t packets_per_chunk = m_media_settings.packets_in_media_unit / chunk_cnt;
            if (packets_per_chunk <= TARGET_PACKETS_PER_CHUNK) {
                m_media_settings.packets_in_chunk = packets_per_chunk;
                break;
            }
        }

        std::cout << "Calculated audio chunk size: " << m_media_settings.packets_in_chunk
                  << " packets per chunk ("
                  << (m_media_settings.packets_in_media_unit / m_media_settings.packets_in_chunk)
                  << " chunks per media unit)" << std::endl;
    }

    // Calculate chunks per media unit
    m_media_settings.chunks_in_media_unit =
        m_media_settings.packets_in_media_unit / m_media_settings.packets_in_chunk;

    return ReturnStatus::success;
}

ReturnStatus ST_2110_30_MediaSettingsCalculator::calculate_timing_parameters()
{
    // Audio uses sampling rate as RTP clock rate (per ST 2110-30)
    auto rate_it = AUDIO_SAMPLING_RATE_MAP.find(m_media_settings.sampling_rate);
    if (rate_it == AUDIO_SAMPLING_RATE_MAP.end()) {
        std::cerr << "Error: Unsupported audio sampling rate: "
                  << static_cast<int>(m_media_settings.sampling_rate) << std::endl;
        return ReturnStatus::failure;
    }
    m_media_settings.sample_rate = rate_it->second;
    m_media_settings.media_unit_time_interval_ns = static_cast<double>(m_media_settings.packets_in_media_unit * m_media_settings.ptime_usec * NS_IN_USEC);
    m_media_settings.ticks_per_media_unit = Rational(
        m_media_settings.sample_rate * m_media_settings.packets_in_media_unit * m_media_settings.ptime_usec,
        USEC_IN_SEC);

    return ReturnStatus::success;
}

void ST_2110_30_MediaSettingsCalculator::calculate_memory_parameters()
{
    if (m_media_settings.media_units_in_mem_block == 0) {
        m_media_settings.media_units_in_mem_block = m_media_settings.DEFAULT_NUM_OF_MEDIA_UNITS_IN_MEM_BLOCK;
    }

    m_media_settings.chunks_in_mem_block = m_media_settings.media_units_in_mem_block * m_media_settings.chunks_in_media_unit;
    m_media_settings.packets_in_mem_block = m_media_settings.chunks_in_mem_block * m_media_settings.packets_in_chunk;
    m_media_settings.bytes_per_media_unit = m_media_settings.raw_packet_payload_size * m_media_settings.packets_in_media_unit;
}

void ST_2110_30_MediaSettingsCalculator::calculate_stride_parameters()
{
    m_media_settings.app_header_stride_size = align_up_pow2(m_media_settings.packet_app_header_size, get_cache_line_size());
    m_media_settings.data_stride_size = align_up_pow2(m_media_settings.packet_payload_size, get_cache_line_size());
}

ReturnStatus ST_2110_30_MediaSettingsCalculator::calculate_media_settings()
{
    if (!is_channel_count_supported(m_media_settings.num_channels)) {
        std::cerr << "Unsupported channel count: " << static_cast<uint32_t>(m_media_settings.num_channels)
                  << " channels." << std::endl;
        return ReturnStatus::failure;
    }

    ReturnStatus status = calculate_packet_parameters();
    if (status != ReturnStatus::success) {
        return status;
    }

    status = calculate_timing_parameters();
    if (status != ReturnStatus::success) {
        return status;
    }

    calculate_memory_parameters();
    calculate_stride_parameters();

    return ReturnStatus::success;
}

std::string ST_2110_30_MediaSettingsCalculator::generate_media_sdp(const std::vector<NetworkFlow>& flows)
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
        .set_session_name("SMPTE ST2110-30");
    if (is_multi_flow) {
        session_builder.add_group(GroupAttribute::Builder("DUP", group_mids).build());
    }

    auto time_description = TimeDescription::Builder().build();
    auto sdp_builder = SDPManager::Builder(session_builder.build(), std::move(time_description));

    for (size_t i = 0; i < flows.size(); ++i) {
        const auto& flow = flows[i];

        SMPTE2110_30_MediaDescription::Builder media_builder {
            flow.destination_port, TransportProtocol::RTP_AVP, std::to_string(m_media_settings.payload_type), flow.destination_ip
        };
        media_builder
            .set_source_filter(SourceFilterAttribute::Builder(flow.destination_ip, flow.source_ip).build())
            .set_encoding(m_media_settings.encoding)
            .set_sampling_rate(m_media_settings.sampling_rate)
            .set_channels(m_media_settings.num_channels)
            .set_ptime(m_media_settings.ptime_usec / 1000.0)
            .set_timestamp_ref_clock(
                m_media_settings.ref_clk_is_ptp ? TimestampRefClock::PTP : TimestampRefClock::LocalMAC
            )
            .set_timestamp_ref_clock_ptp_traceable(
                m_media_settings.ref_clk_is_ptp && m_media_settings.refclk_id.empty()
            )
            .set_timestamp_ref_clock_local_mac(m_media_settings.refclk_id)
            .set_extra_format_specific_parameters(m_extra_parameters);

        if (is_multi_flow) {
            media_builder.set_media_id(std::string(1, 'a' + static_cast<char>(i)));
        }

        sdp_builder.add_media_description(media_builder.build());
    }

    return sdp_builder.build()->to_string();
}

std::string ST_2110_30_MediaSettingsCalculator::get_smpte_standard_name() const
{
    return "Audio";
}

double ST_2110_30_MediaSettingsCalculator::align_time_to_media_unit_boundary_ns(uint64_t desired_time_ns) const
{
    double media_unit_interval_ns = m_media_settings.media_unit_time_interval_ns;

    // Find the next aligned media unit start time
    uint64_t N = static_cast<uint64_t>(static_cast<double>(desired_time_ns) / media_unit_interval_ns + 1);
    double first_packet_start_time_ns = N * media_unit_interval_ns;

    return first_packet_start_time_ns;
}
