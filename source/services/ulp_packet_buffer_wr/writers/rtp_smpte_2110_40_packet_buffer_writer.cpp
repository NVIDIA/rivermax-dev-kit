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

#include <cassert>
#include <cstddef>
#include <cstring>

#include "rdk/services/ulp_packet_buffer_wr/writers/rtp_smpte_2110_40_packet_buffer_writer.h"

using namespace rivermax::dev_kit::services;

RTP_SMPTE_2110_40_PacketBufferWriter::RTP_SMPTE_2110_40_PacketBufferWriter(const MediaSettings& media_settings,
    std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils, bool enable_mock_mode)
    : RTPMediaPacketBufferWriter<RTP_SMPTE_2110_40_PacketContext, RTP_SMPTE_2110_40_Packet>(
        media_settings, std::move(header_mem_utils), std::move(payload_mem_utils), enable_mock_mode)
{
    if (enable_mock_mode) {
        m_rtp_packet = std::make_unique<RTP_SMPTE_2110_40_MockPacket>(nullptr, nullptr);
    }
    auto& ancillary_settings = static_cast<const SMPTE_2110_40_MediaSettings&>(m_media_settings);
    // TODO: Support multiple ancillary packets in a single RTP packet.
    // These fields should be calculated based on media unit data / metadata.
    m_rtp_packet_context->did = ancillary_settings.did;
    m_rtp_packet_context->sdid = ancillary_settings.sdid;
    m_rtp_packet_context->user_data_words_count = ancillary_settings.user_data_words_count;
}

void RTP_SMPTE_2110_40_PacketBufferWriter::reset_in_media_unit_state()
{
    m_rtp_packet_context->counter = 0;
    m_rtp_packet_context->field_indicator = 0;
}

void RTP_SMPTE_2110_40_PacketBufferWriter::update_in_media_unit_state(size_t header_size, size_t payload_size)
{
    auto& ancillary_settings = static_cast<const SMPTE_2110_40_MediaSettings&>(m_media_settings);
    // ST 2110-40: timestamp is the same for all packets in a frame (like video)
    // Only increment after the last packet of the frame
    if (++m_rtp_packet_context->counter >= m_media_settings.packets_in_media_unit) {
        // Timestamp changes every frame (90kHz clock)
        m_rtp_packet_context->timestamp += static_cast<uint32_t>(m_media_settings.ticks_per_media_unit);
        m_rtp_packet_context->counter = 0;

        // Toggle field indicator for interlaced content
        // Note: This is simplified; real implementation should check video_scan_type
        m_rtp_packet_context->field_indicator = (m_rtp_packet_context->field_indicator == 0) ? 1 : 0;
    }
    // Set Marker bit on last ANC data RTP packet for a field (for interlaced video).
    m_rtp_packet_context->marker = (m_rtp_packet_context->counter == ancillary_settings.packets_in_media_unit - 1) ? 1 : 0;
    m_rtp_packet_context->sequence++;
    m_rtp_packet_context->extended_sequence_number++;
}

ReturnStatus RTP_SMPTE_2110_40_PacketBufferWriter::write_buffer(void* header_ptr, void* payload_ptr, size_t buffer_length)
{
    byte_t* current_header_pointer = reinterpret_cast<byte_t*>(header_ptr);
    byte_t* current_payload_pointer = reinterpret_cast<byte_t*>(payload_ptr);
    assert(current_header_pointer);
    assert(current_payload_pointer);
    uint64_t stride = 0;
    size_t header_size = 0;
    size_t payload_size = 0;

    while (stride < buffer_length && m_rtp_packet_context->counter < m_media_settings.packets_in_media_unit) {
        m_rtp_packet->set_packet(current_header_pointer, current_payload_pointer);
        (void)m_rtp_packet->fill_header(*m_rtp_packet_context, header_size, m_header_mem_utils.get());
        (void)m_rtp_packet->fill_payload(*m_rtp_packet_context, payload_size, m_payload_mem_utils.get());
        update_in_media_unit_state(header_size, payload_size);
        current_header_pointer += m_media_settings.app_header_stride_size;
        current_payload_pointer += m_media_settings.data_stride_size;
        stride++;
    }
    return ReturnStatus::success;
}
