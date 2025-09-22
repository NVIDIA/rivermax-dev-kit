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

#include <cstdint>
#include <iostream>

#include "rdk/services/media/ancillary_settings_calculator.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/utils/defs.h"
#include "rt_threads.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

//todo t2 : use def from sdp once merged


ReturnStatus ST_2110_40_MediaSettingsCalculator::calculate_packet_parameters()
{
    // Calculate actual packet size
    m_media_settings.protocol_header_size = RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE;
    m_media_settings.raw_packet_payload_size = SMPTE_2110_40_MediaSettings::ANCILLARY_DATA_HEADER_SIZE + m_media_settings.user_data_size_bytes;
    m_media_settings.packet_payload_size = m_media_settings.protocol_header_size + m_media_settings.raw_packet_payload_size;
    
    // Packet size validation
    if (m_media_settings.packet_payload_size > MediaSettings::MAX_PAYLOAD_SIZE) {
        std::cerr << "Error: Ancillary packet size (" << m_media_settings.packet_payload_size 
                  << " bytes) exceeds network limit (" << MediaSettings::MAX_PAYLOAD_SIZE << " bytes). "
                  << "Consider smaller user_data_size_bytes." << std::endl;
        return ReturnStatus::failure;
    }
    
    // HDS for ANC keeps Anc Data Header with payload
    if (m_media_settings.header_data_split) {
        m_media_settings.packet_app_header_size = m_media_settings.protocol_header_size;
        m_media_settings.packet_payload_size -= m_media_settings.protocol_header_size;
    }
    
    // Default to 1 packet for simple anc if not provided
    if (m_media_settings.packets_in_frame_field == 0) {
        m_media_settings.packets_in_frame_field = 1;
    }

    // Validate and apply custom packets in chunk if provided
    constexpr uint32_t DEFAULT_PACKETS_PER_CHUNK = 1;
    if (m_media_settings.packets_in_chunk > 0) {
        if (m_media_settings.packets_in_frame_field % m_media_settings.packets_in_chunk == 0) {
            std::cout << "Using custom ancillary chunk size: " << m_media_settings.packets_in_chunk
                      << " packets per chunk" << std::endl;
        } else {
            std::cerr << "Warning: Custom chunk size (" << m_media_settings.packets_in_chunk
                      << ") is not a divisor of packets in field ("
                      << m_media_settings.packets_in_frame_field << "). Using default (1)." << std::endl;
            m_media_settings.packets_in_chunk = DEFAULT_PACKETS_PER_CHUNK;
        }
    } else {
        // Default: 1 packet per chunk for simple ancillary
        m_media_settings.packets_in_chunk = DEFAULT_PACKETS_PER_CHUNK;
    }

    // Calculate chunks per frame field
    m_media_settings.chunks_in_frame_field =
        m_media_settings.packets_in_frame_field / m_media_settings.packets_in_chunk;
    
    return ReturnStatus::success;
}

void ST_2110_40_MediaSettingsCalculator::calculate_timing_parameters()
{
    m_media_settings.sample_rate = MediaSettings::RTP_SAMPLE_RATE;
    double fps_value = static_cast<double>(m_media_settings.frame_rate.num) / m_media_settings.frame_rate.denom;
    m_media_settings.frame_field_time_interval_ns = NS_IN_SEC / fps_value;
    m_media_settings.ticks_per_frame = static_cast<double>(m_media_settings.sample_rate) / fps_value;
}

void ST_2110_40_MediaSettingsCalculator::calculate_memory_parameters()
{
    if (m_media_settings.frames_fields_in_mem_block == 0) {
        m_media_settings.frames_fields_in_mem_block = m_media_settings.DEFAULT_NUM_OF_FRAMES_IN_MEM_BLOCK;
    }
    m_media_settings.chunks_in_mem_block = m_media_settings.frames_fields_in_mem_block * m_media_settings.chunks_in_frame_field;
    m_media_settings.packets_in_mem_block = m_media_settings.chunks_in_mem_block * m_media_settings.packets_in_chunk;
    m_media_settings.bytes_per_frame = m_media_settings.raw_packet_payload_size * m_media_settings.packets_in_frame_field;
}

void ST_2110_40_MediaSettingsCalculator::calculate_stride_parameters()
{
    m_media_settings.app_header_stride_size = align_up_pow2(m_media_settings.packet_app_header_size, get_cache_line_size());
    m_media_settings.data_stride_size = align_up_pow2(m_media_settings.packet_payload_size, get_cache_line_size());
}

ReturnStatus ST_2110_40_MediaSettingsCalculator::calculate_media_settings()
{
    // Validate that user provided required DID/SDID values
    if (m_media_settings.did == 0 || m_media_settings.sdid == 0) {
        std::cerr << "Error: DID (" << m_media_settings.did << ") and SDID (" << m_media_settings.sdid 
                  << ") must be provided (non-zero)." << std::endl;
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

std::string ST_2110_40_MediaSettingsCalculator::generate_media_sdp(
    const std::string& source_ip, const uint16_t source_port,
    const std::string& destination_ip, const uint16_t destination_port)
{
    std::string sdp_stub;
    return sdp_stub;
}

std::string ST_2110_40_MediaSettingsCalculator::get_media_type_name() const
{
    return "Ancillary";
}

} // namespace services
} // namespace dev_kit
} // namespace rivermax
