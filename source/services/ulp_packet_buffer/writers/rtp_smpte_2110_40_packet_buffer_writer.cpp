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

#include <cassert>
#include <cstddef>
#include <cstring>

#include "rdk/services/ulp_packet_buffer/writers/rtp_smpte_2110_40_packet_buffer_writer.h"
#include "rdk/services/ulp_packet/smpte_2110_40_ancillary_utils.h"

using namespace rdk::services;

RTP_SMPTE_2110_40_PacketBufferWriter::RTP_SMPTE_2110_40_PacketBufferWriter(const MediaSettings& media_settings,
    std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils, bool enable_zero_copy)
    : RTPMediaPacketBufferWriter<RTP_SMPTE_2110_40_PacketContext, RTP_SMPTE_2110_40_PacketWriter, AncillaryMediaUnitMetadata>(
        media_settings, std::move(header_mem_utils), std::move(payload_mem_utils), enable_zero_copy)
{
    if (enable_zero_copy) {
        m_cached_packets_in_media_unit = calculate_packets_for_media_unit();
    }
    const auto& ancillary_settings = static_cast<const SMPTE_2110_40_MediaSettings&>(media_settings);
    m_current_field_indicator =
        (ancillary_settings.video_scan_type == VideoScanType::Interlaced) ? RTP_2110_40_FIELD_INDICATOR_FIELD1
                                                                         : RTP_2110_40_FIELD_INDICATOR_PROGRESSIVE;
}

ReturnStatus RTP_SMPTE_2110_40_PacketBufferWriter::set_next_media_unit(std::shared_ptr<MediaUnit> media_unit)
{
    ReturnStatus status = RTPMediaPacketBufferWriter::set_next_media_unit(std::move(media_unit));
    if (status != ReturnStatus::success) {
        return status;
    }

    // Extract descriptors pointer from metadata
    if (m_current_media_unit && m_current_media_unit->metadata) {
        auto* ancillary_metadata = static_cast<AncillaryMediaUnitMetadata*>(m_current_media_unit->metadata.get());
        m_descriptors = &ancillary_metadata->ancillary_data;
    } else {
        m_descriptors = nullptr;
    }

    m_cached_packets_in_media_unit = calculate_packets_for_media_unit();

    return ReturnStatus::success;
}

void RTP_SMPTE_2110_40_PacketBufferWriter::reset_in_media_unit_state()
{
    m_current_descriptor_index = 0;
}

void RTP_SMPTE_2110_40_PacketBufferWriter::prepare_context_for_packet()
{
    const auto& ancillary_settings = static_cast<const SMPTE_2110_40_MediaSettings&>(m_media_settings);

    // Set source data pointer for payload (nullptr in zero copy mode)
    if (!m_zero_copy_enabled && m_current_media_unit) {
        m_rtp_packet_context->payload_ptr = m_current_media_unit->data->get();
    } else {
        m_rtp_packet_context->payload_ptr = nullptr;
    }

    m_rtp_packet_context->descriptors = m_descriptors;
    m_rtp_packet_context->descriptor_start_index = m_current_descriptor_index;
    m_rtp_packet_context->field_indicator = m_current_field_indicator;

    // Calculate how many descriptors fit in this packet
    if (m_descriptors) {
        m_rtp_packet_context->descriptor_count_in_packet = calculate_descriptors_in_packet(
            *m_descriptors,
            m_current_descriptor_index,
            m_media_settings.raw_packet_payload_size,
            ancillary_settings.max_ancillary_data_packets_per_packet);
    } else {
        m_rtp_packet_context->descriptor_count_in_packet = 0;
    }

    // Set marker bit based on whether this is the last packet
    m_rtp_packet_context->marker = (m_packet_counter == m_cached_packets_in_media_unit - 1) ? 1 : 0;
}

size_t RTP_SMPTE_2110_40_PacketBufferWriter::calculate_descriptors_in_packet(
    const std::vector<AncillaryDataDescriptor>& descriptors,
    size_t start_index,
    size_t max_payload_size,
    size_t max_ancillary_data_packets_per_packet)
{
    if (start_index >= descriptors.size()) {
        return 0;
    }

    size_t current_payload_size = 0;
    size_t descriptor_count = 0;

    for (size_t i = start_index; i < descriptors.size(); ++i) {
        const size_t ancillary_data_packet_size = AncillaryDataUtils::calculate_packet_size(
            descriptors[i].ancillary_data_header.user_data_words_count);

        // Make sure it doesn't exceed the max number of ancillary data packets per RTP packet
        if (descriptor_count >= max_ancillary_data_packets_per_packet) {
            break;
        }
        // Make sure it doesn't exceed the max payload size
        if (current_payload_size + ancillary_data_packet_size > max_payload_size) {
            break;
        }

        current_payload_size += ancillary_data_packet_size;
        descriptor_count++;
    }

    return descriptor_count;
}

size_t RTP_SMPTE_2110_40_PacketBufferWriter::calculate_rtp_packets_for_descriptors(
    const std::vector<AncillaryDataDescriptor>& descriptors,
    size_t max_payload_size,
    size_t max_ancillary_data_packets_per_packet)
{
    // At least one packet is needed for marker bit
    if (descriptors.empty()) {
        return 1;
    }

    size_t rtp_packet_count = 0;
    size_t current_index = 0;
    while (current_index < descriptors.size()) {
        size_t descriptors_in_packet = calculate_descriptors_in_packet(descriptors, current_index, max_payload_size, max_ancillary_data_packets_per_packet);
        if (descriptors_in_packet == 0) {
            break;
        }
        current_index += descriptors_in_packet;
        rtp_packet_count++;
    }

    return rtp_packet_count;
}

size_t RTP_SMPTE_2110_40_PacketBufferWriter::calculate_packets_for_media_unit() const
{
    size_t data_packets_needed;
    if (!m_descriptors) {
        data_packets_needed = m_media_settings.packets_in_media_unit;
    } else {
        const auto& ancillary_settings = static_cast<const SMPTE_2110_40_MediaSettings&>(m_media_settings);
        data_packets_needed = calculate_rtp_packets_for_descriptors(
            *m_descriptors,
            m_media_settings.raw_packet_payload_size,
            ancillary_settings.max_ancillary_data_packets_per_packet);
    }

    size_t chunks_needed_for_data = (data_packets_needed + m_media_settings.packets_in_chunk - 1) /
                                        m_media_settings.packets_in_chunk;
    size_t empty_chunks = 0;
    if (chunks_needed_for_data < m_media_settings.chunks_in_media_unit) {
        empty_chunks = m_media_settings.chunks_in_media_unit - chunks_needed_for_data;
    }
    return data_packets_needed + empty_chunks;
}

void RTP_SMPTE_2110_40_PacketBufferWriter::update_in_media_unit_state(size_t header_size, size_t payload_size)
{
    // Advance descriptor index by number of descriptors written in this packet
    if (m_descriptors) {
        m_current_descriptor_index += m_rtp_packet_context->descriptor_count_in_packet;
    }

    m_packet_counter++;

    // ST 2110-40: timestamp is the same for all packets in a frame (like video)
    // Only increment after the last packet of the frame
    if (m_packet_counter >= m_cached_packets_in_media_unit) {
        // Timestamp changes every frame (90kHz clock)
        m_rtp_packet_context->timestamp += m_media_settings.ticks_per_media_unit;
        m_packet_counter = 0;
        m_current_descriptor_index = 0;

        // Toggle field indicator for interlaced content
        const auto& ancillary_settings = static_cast<const SMPTE_2110_40_MediaSettings&>(m_media_settings);
        if (ancillary_settings.video_scan_type == VideoScanType::Interlaced) {
            m_current_field_indicator =
                (m_current_field_indicator == RTP_2110_40_FIELD_INDICATOR_FIELD1)
                    ? RTP_2110_40_FIELD_INDICATOR_FIELD2
                    : RTP_2110_40_FIELD_INDICATOR_FIELD1;
        }
    }
    m_rtp_packet_context->sequence++;
    m_rtp_packet_context->extended_sequence_number++;
}

ReturnStatus RTP_SMPTE_2110_40_PacketBufferWriter::write_buffer(void* payload_ptr, size_t buffer_length, uint16_t* payload_sizes)
{
    byte_t* current_packet_pointer = reinterpret_cast<byte_t*>(payload_ptr);
    assert(current_packet_pointer);
    assert(payload_sizes);
    uint64_t stride = 0;
    size_t header_size = 0;
    size_t payload_size = 0;

    while (stride < buffer_length && m_packet_counter < m_cached_packets_in_media_unit) {
        m_rtp_packet_writer->set_packet(current_packet_pointer);
        prepare_context_for_packet();
        // Skip ReturnStatus testing for performance reasons
        (void)m_rtp_packet_writer->fill_header(*m_rtp_packet_context, header_size, m_header_mem_utils.get());
        (void)m_rtp_packet_writer->fill_payload(*m_rtp_packet_context, payload_size, m_payload_mem_utils.get());
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

    while (stride < buffer_length && m_packet_counter < m_cached_packets_in_media_unit) {
        m_rtp_packet_writer->set_packet(current_header_pointer, current_payload_pointer);
        prepare_context_for_packet();
        (void)m_rtp_packet_writer->fill_header(*m_rtp_packet_context, header_size, m_header_mem_utils.get());
        (void)m_rtp_packet_writer->fill_payload(*m_rtp_packet_context, payload_size, m_payload_mem_utils.get());
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
    // When no descriptors (e.g. zero copy mode or no metadata), use default packets per chunk
    if (!m_descriptors) {
        return RTPMediaPacketBufferWriter::get_num_packets_for_next_chunk();
    }

    size_t total_descriptors = m_descriptors->size();

    if (m_current_descriptor_index >= total_descriptors) {
        return 1;
    }

    // Create a view of remaining descriptors
    std::vector<AncillaryDataDescriptor> remaining_descriptors(
        m_descriptors->begin() + m_current_descriptor_index,
        m_descriptors->end());

    // Calculate RTP packets needed for remaining descriptors
    const auto& ancillary_settings = static_cast<const SMPTE_2110_40_MediaSettings&>(m_media_settings);
    size_t packets_for_remaining_descriptors = calculate_rtp_packets_for_descriptors(
        remaining_descriptors,
        m_media_settings.raw_packet_payload_size,
        ancillary_settings.max_ancillary_data_packets_per_packet);

    // Also check how many packets remain in the media unit (accounts for empty marker packets)
    size_t packets_remaining_in_media_unit = (m_packet_counter < m_cached_packets_in_media_unit)
        ? (m_cached_packets_in_media_unit - m_packet_counter) : 0;

    size_t packets_available = std::min(packets_for_remaining_descriptors, packets_remaining_in_media_unit);
    size_t packets_for_this_chunk = std::min(packets_available, m_media_settings.packets_in_chunk);

    // Return at least 1 packet per chunk for marker bit in empty chunks
    return std::max(packets_for_this_chunk, static_cast<size_t>(1));
}
