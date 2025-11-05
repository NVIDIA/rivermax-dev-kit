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
    : RTPMediaPacketBufferWriter<RTP_SMPTE_2110_40_PacketContext, RTP_SMPTE_2110_40_Packet, AncillaryMediaUnitMetadata>(
        media_settings, std::move(header_mem_utils), std::move(payload_mem_utils), enable_mock_mode)
{
    if (enable_mock_mode) {
        const auto& ancillary_settings = static_cast<const SMPTE_2110_40_MediaSettings&>(media_settings);
        m_rtp_packet = std::make_unique<RTP_SMPTE_2110_40_MockPacket>(nullptr, nullptr);
        m_rtp_packet_context->ancillary_data_descriptor.ancillary_data_header.did = ancillary_settings.data_identifiers[m_rtp_packet_context->counter].did;
        m_rtp_packet_context->ancillary_data_descriptor.ancillary_data_header.sdid = ancillary_settings.data_identifiers[m_rtp_packet_context->counter].sdid;
        m_rtp_packet_context->ancillary_data_descriptor.ancillary_data_header.user_data_words_count = ancillary_settings.max_user_data_words_count;
    }
}

ReturnStatus RTP_SMPTE_2110_40_PacketBufferWriter::set_next_media_unit(std::shared_ptr<MediaUnit> media_unit)
{
    ReturnStatus status = RTPMediaPacketBufferWriter::set_next_media_unit(std::move(media_unit));
    if (status != ReturnStatus::success) {
        return status;
    }

    if (!m_metadata_ptr->ancillary_data.empty()) {
        m_rtp_packet_context->ancillary_data_descriptor = m_metadata_ptr->ancillary_data[0];
    }

    return ReturnStatus::success;
}

void RTP_SMPTE_2110_40_PacketBufferWriter::reset_in_media_unit_state()
{
    m_rtp_packet_context->counter = 0;
    m_rtp_packet_context->field_indicator = 0;
}

size_t RTP_SMPTE_2110_40_PacketBufferWriter::get_num_packets_for_media_unit() const
{
    size_t total_ancillary_packets = 0;

    // Use calculated settings in mock mode
    if (m_mock_mode_enabled || !m_metadata_ptr) {
        total_ancillary_packets = m_media_settings.packets_in_media_unit;
    } else {
        total_ancillary_packets = m_metadata_ptr->ancillary_data.size();
    }

    size_t packets_per_chunk = std::max(
        (total_ancillary_packets + m_media_settings.chunks_in_media_unit - 1) / m_media_settings.chunks_in_media_unit,
        static_cast<size_t>(1));
    size_t packets_before_last_chunk = packets_per_chunk * (m_media_settings.chunks_in_media_unit - 1);

    // Last chunk might not need any packets. Make sure we have at least 1.
    size_t remaining_ancillary = (total_ancillary_packets > packets_before_last_chunk) ? (total_ancillary_packets - packets_before_last_chunk) : 0;
    size_t packets_in_last_chunk = std::max(remaining_ancillary, static_cast<size_t>(1));
    return packets_before_last_chunk + packets_in_last_chunk;
}

void RTP_SMPTE_2110_40_PacketBufferWriter::update_in_media_unit_state(size_t header_size, size_t payload_size)
{
    size_t total_packets_in_media_unit = get_num_packets_for_media_unit();

    // ST 2110-40: timestamp is the same for all packets in a frame (like video)
    // Only increment after the last packet of the frame
    if (++m_rtp_packet_context->counter >= total_packets_in_media_unit) {
        // Timestamp changes every frame (90kHz clock)
        m_rtp_packet_context->timestamp += static_cast<uint32_t>(m_media_settings.ticks_per_media_unit);
        m_rtp_packet_context->counter = 0;

        // Toggle field indicator for interlaced content
        // Note: This is simplified; real implementation should check video_scan_type
        m_rtp_packet_context->field_indicator = (m_rtp_packet_context->field_indicator == 0) ? 1 : 0;
    }
    // Set Marker bit on last ANC data RTP packet for a field (for interlaced video).
    m_rtp_packet_context->marker = (m_rtp_packet_context->counter == total_packets_in_media_unit - 1) ? 1 : 0;
    m_rtp_packet_context->sequence++;
    m_rtp_packet_context->extended_sequence_number++;

    // Update ancillary data header for next packet if anc packets left in media unit
    if (!m_mock_mode_enabled) {
        if (m_rtp_packet_context->counter < m_metadata_ptr->ancillary_data.size()) {
            m_rtp_packet_context->ancillary_data_descriptor = m_metadata_ptr->ancillary_data[m_rtp_packet_context->counter];
        } else {
            m_rtp_packet_context->ancillary_data_descriptor = AncillaryDataDescriptor{};
        }
    }
}

ReturnStatus RTP_SMPTE_2110_40_PacketBufferWriter::write_buffer(void* payload_ptr, size_t buffer_length, uint16_t* payload_sizes)
{
    byte_t* current_packet_pointer = reinterpret_cast<byte_t*>(payload_ptr);
    assert(current_packet_pointer);
    assert(payload_sizes);
    uint64_t stride = 0;
    size_t header_size = 0;
    size_t payload_size = 0;
    size_t total_packets_in_media_unit = get_num_packets_for_media_unit();

    while (stride < buffer_length && m_rtp_packet_context->counter < total_packets_in_media_unit) {
        m_rtp_packet->set_packet(current_packet_pointer);
        // Skip ReturnStatus testing for performance reasons
        (void)m_rtp_packet->fill_header(*m_rtp_packet_context, header_size, m_header_mem_utils.get());
        (void)m_rtp_packet->fill_payload(*m_rtp_packet_context, payload_size, m_payload_mem_utils.get());
        payload_sizes[stride] = static_cast<uint16_t>(header_size + payload_size);
        update_in_media_unit_state(header_size, payload_size);
        current_packet_pointer += m_media_settings.data_stride_size;
        stride++;
    }
    return ReturnStatus::success;
}

ReturnStatus RTP_SMPTE_2110_40_PacketBufferWriter::write_buffer(void* header_ptr, void* payload_ptr, size_t buffer_length, uint16_t* header_sizes, uint16_t* payload_sizes)
{
    byte_t* current_header_pointer = reinterpret_cast<byte_t*>(header_ptr);
    byte_t* current_payload_pointer = reinterpret_cast<byte_t*>(payload_ptr);
    assert(current_header_pointer);
    assert(current_payload_pointer);
    assert(header_sizes);
    assert(payload_sizes);
    uint64_t stride = 0;
    size_t header_size = 0;
    size_t payload_size = 0;
    size_t total_packets_in_media_unit = get_num_packets_for_media_unit();

    while (stride < buffer_length && m_rtp_packet_context->counter < total_packets_in_media_unit) {
        m_rtp_packet->set_packet(current_header_pointer, current_payload_pointer);
        (void)m_rtp_packet->fill_header(*m_rtp_packet_context, header_size, m_header_mem_utils.get());
        (void)m_rtp_packet->fill_payload(*m_rtp_packet_context, payload_size, m_payload_mem_utils.get());
        header_sizes[stride] = static_cast<uint16_t>(header_size);
        payload_sizes[stride] = static_cast<uint16_t>(payload_size);
        update_in_media_unit_state(header_size, payload_size);
        current_header_pointer += m_media_settings.app_header_stride_size;
        current_payload_pointer += m_media_settings.data_stride_size;
        stride++;
    }
    return ReturnStatus::success;
}

size_t RTP_SMPTE_2110_40_PacketBufferWriter::get_num_packets_for_next_chunk() const
{
    // In mock mode, use default packets per chunk
    if (m_mock_mode_enabled || !m_metadata_ptr) {
        return RTPMediaPacketBufferWriter::get_num_packets_for_next_chunk();
    }

    size_t total_ancillary_packets = m_metadata_ptr->ancillary_data.size();
    size_t packets_written = m_rtp_packet_context->counter;
    size_t packets_per_chunk = std::max(
        (total_ancillary_packets + m_media_settings.chunks_in_media_unit - 1) / m_media_settings.chunks_in_media_unit,
        static_cast<size_t>(1));
    size_t chunks_written = packets_written / packets_per_chunk;
    size_t remaining_chunks = m_media_settings.chunks_in_media_unit - chunks_written;

    // For the last chunk, return at least 1 packet if none left
    if (remaining_chunks == 1) {
        size_t remaining_ancillary = (total_ancillary_packets > packets_written) ? (total_ancillary_packets - packets_written) : 0;
        return std::max(remaining_ancillary, static_cast<size_t>(1));
    }
    return packets_per_chunk;
}
