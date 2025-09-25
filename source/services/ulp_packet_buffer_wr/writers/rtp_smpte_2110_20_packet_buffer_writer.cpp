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

#include "rdk/services/ulp_packet_buffer_wr/writers/rtp_smpte_2110_20_packet_buffer_writer.h"
#include "rdk/services/media/media_settings_video.h"

using namespace rivermax::dev_kit::services;

RTP_SMPTE_2110_20_MockPacketBufferWriter::RTP_SMPTE_2110_20_MockPacketBufferWriter(const MediaSettings& media_settings,
    std::shared_ptr<MemoryUtils> header_mem_utils, 
    std::shared_ptr<MemoryUtils> payload_mem_utils)
    : RTPMediaPacketBufferWriter<RTP_SMPTE_2110_20_PacketContext>(media_settings, std::move(header_mem_utils), std::move(payload_mem_utils))
{
    m_rtp_packet_context->srd_length = media_settings.raw_packet_payload_size;
}

ReturnStatus RTP_SMPTE_2110_20_MockPacketBufferWriter::set_next_media_unit(std::shared_ptr<MediaUnit> media_unit)
{
    reset_in_media_unit_state();
    return ReturnStatus::success;
}

void RTP_SMPTE_2110_20_MockPacketBufferWriter::reset_in_media_unit_state()
{
    m_rtp_packet_context->counter = 0;
    m_rtp_packet_context->line_number = 0;
    m_rtp_packet_context->srd_offset = 0;
    m_rtp_packet_context->rtp_interlace_field_indicator = 0;
}

inline void RTP_SMPTE_2110_20_MockPacketBufferWriter::update_in_media_unit_state(size_t header_size, size_t payload_size)
{
    auto& video_settings = static_cast<const SMPTE_2110_20_MediaSettings&>(m_media_settings);
    m_rtp_packet_context->srd_length = video_settings.raw_packet_payload_size;
    m_rtp_packet_context->srd_offset = (m_rtp_packet_context->srd_offset + video_settings.pixels_per_packet) %
        (video_settings.resolution.width);

    if (!(++m_rtp_packet_context->counter % video_settings.packets_in_line)) {
        // Prepare line number for next iteration:
        m_rtp_packet_context->line_number = (m_rtp_packet_context->line_number + 1) % video_settings.lines_in_frame_field;
    }

    if (m_rtp_packet_context->counter == video_settings.packets_in_media_unit) {
        // ST2210-20: the timestamp SHOULD be the same for each packet of the frame/field.
        m_rtp_packet_context->timestamp += static_cast<uint32_t>(video_settings.ticks_per_media_unit);
        m_rtp_packet_context->counter = 0;
        if (video_settings.video_scan_type == VideoScanType::Interlaced) {
            m_rtp_packet_context->rtp_interlace_field_indicator = !m_rtp_packet_context->rtp_interlace_field_indicator;
        }
    }
    m_rtp_packet_context->marker = (m_rtp_packet_context->counter == video_settings.packets_in_media_unit - 1) ? 1 : 0;
    m_rtp_packet_context->sequence++;
    m_rtp_packet_context->extended_sequence_number++;

    m_rtp_packet_context->data_left_in_media_unit_in_bytes -= payload_size;
    if (m_rtp_packet_context->data_left_in_media_unit_in_bytes == 0) {
        m_rtp_packet_context->current_media_unit = nullptr;
    }
}

ReturnStatus RTP_SMPTE_2110_20_PacketBufferWriter::set_next_media_unit(std::shared_ptr<MediaUnit> media_unit)
{
    if (media_unit == nullptr || media_unit->data == nullptr) {
        std::cerr << "Error: Media unit is null or media unit data is null" << std::endl;
        return ReturnStatus::failure;
    }
    reset_in_media_unit_state();
    m_rtp_packet_context->current_media_unit = std::move(media_unit);
    m_rtp_packet_context->data_left_in_media_unit_in_bytes = m_rtp_packet_context->current_media_unit->data->get_size();
    return ReturnStatus::success;
}

ReturnStatus RTP_SMPTE_2110_20_PacketBufferWriter::write_buffer(void* header_ptr, void* payload_ptr, size_t length_in_strides)
{
    byte_t* current_header_pointer = reinterpret_cast<byte_t*>(header_ptr);
    byte_t* current_payload_pointer = reinterpret_cast<byte_t*>(payload_ptr);
    assert(current_header_pointer);
    assert(current_payload_pointer);
    uint64_t stride = 0;
    ReturnStatus status = ReturnStatus::success;

    // Determine how many complete packets we can process
    size_t packets_to_process = std::min(length_in_strides,
        static_cast<size_t>(m_media_settings.packets_in_media_unit - m_rtp_packet_context->counter));
    // Limit by available frame data
    size_t max_packets_by_data =
        (m_rtp_packet_context->data_left_in_media_unit_in_bytes + m_media_settings.raw_packet_payload_size - 1) /
        m_media_settings.raw_packet_payload_size;
    packets_to_process = std::min(packets_to_process, max_packets_by_data);

    if (packets_to_process == 0) {
        std::cerr << "Warning: No packets to process. State may be inconsistent." << std::endl;
        return ReturnStatus::success;
    }

    if (packets_to_process < length_in_strides) {
        std::cerr << "Warning: Requested " << length_in_strides << " packets but only "
                  << packets_to_process << " can be processed due to data constraints." << std::endl;
    }

    size_t header_size = 0;
    for (size_t i = 0; i < packets_to_process; ++i) {
        auto packet = create_packet(current_header_pointer);
        status = packet->fill_header(*m_rtp_packet_context, header_size, m_header_mem_utils.get());
        update_in_media_unit_state(header_size, 0); // Payload size is handled in bulk copy below
        current_header_pointer += m_media_settings.app_header_stride_size;
    }

    byte_t* unit_ptr = m_rtp_packet_context->current_media_unit->data->get() + (m_rtp_packet_context->current_media_unit->data->get_size() - m_rtp_packet_context->data_left_in_media_unit_in_bytes);
    status = m_payload_mem_utils->memory_copy_2D(current_payload_pointer, m_media_settings.data_stride_size,
        unit_ptr, m_media_settings.raw_packet_payload_size,
        m_media_settings.raw_packet_payload_size, packets_to_process, m_rtp_packet_context->current_media_unit->data->get_memory_location());

    size_t data_copied = std::min(packets_to_process * m_media_settings.raw_packet_payload_size, m_rtp_packet_context->data_left_in_media_unit_in_bytes);
    m_rtp_packet_context->data_left_in_media_unit_in_bytes -= data_copied;

    if (status != ReturnStatus::success) {
        std::cerr << "Failed to 2D copy" << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}
