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

#include "rdk/services/buffer_wr/rtp_video_buffer_writer.h"
#include "rdk/services/media/media_settings_2110_20.h"

using namespace rivermax::dev_kit::services;

struct SRDHeader {
    uint16_t srd_length;  // SRD Length: 16 bits

    uint8_t srd_row_number_8_to_14_7bit: 7; // SRD raw number: 15 bits
    uint8_t f: 1;                           // Field identification: 1 bit
    uint8_t srd_row_number_0_to_7_8bit;     // SRD raw number: 15 bits

    uint8_t srd_offset_8_to_14_7bit: 7;     // SRD offset: 15 bits
    uint8_t c: 1;                           // Field identification: 1 bit
    uint8_t srd_offset_0_to_7_8bit;         // SRD offset: 15 bits

    void set_srd_row_number(uint16_t srd_raw_number) {
        srd_row_number_0_to_7_8bit = static_cast<uint8_t>(srd_raw_number);
        srd_row_number_8_to_14_7bit = static_cast<uint8_t>(srd_raw_number >> 8);
    }

    void set_srd_offset(uint16_t srd_offset) {
        srd_offset_0_to_7_8bit = static_cast<uint8_t>(srd_offset);
        srd_offset_8_to_14_7bit = static_cast<uint8_t>(srd_offset >> 8);
    }
 };

 RTPVideoMockBufferWriter::RTPVideoMockBufferWriter(const MediaSettings& media_settings,
    std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils) :
    RTPMediaBufferWriter(media_settings, std::move(header_mem_utils), std::move(payload_mem_utils))
{
    set_stream_properties();
}

ReturnStatus RTPVideoMockBufferWriter::set_next_frame(std::shared_ptr<MediaFrame> frame)
{
    reset_in_frame_state();
    return ReturnStatus::success;
}

void RTPVideoMockBufferWriter::reset_in_frame_state()
{
    m_send_data.packet_counter = 0;
    m_send_data.line_number = 0;
    m_send_data.srd_offset = 0;
    m_send_data.rtp_interlace_field_indicator = 0;
}

inline void RTPVideoMockBufferWriter::update_in_frame_state()
{
    auto& video_settings = static_cast<const SMPTE_2110_20_MediaSettings&>(m_media_settings);
    m_send_data.srd_offset = (m_send_data.srd_offset + video_settings.pixels_per_packet) %
        (video_settings.resolution.width);
    if (!((m_send_data.packet_counter + 1) % video_settings.packets_in_line)) {
        // Prepare line number for next iteration:
        m_send_data.line_number = (m_send_data.line_number + 1) % video_settings.lines_in_frame_field;
    }
    if (++m_send_data.packet_counter == video_settings.packets_in_frame_field) {
        // ST2210-20: the timestamp SHOULD be the same for each packet of the frame/field.
        m_send_data.rtp_timestamp += static_cast<uint32_t>(video_settings.ticks_per_frame);
        m_send_data.packet_counter = 0;
        if (video_settings.video_scan_type == VideoScanType::Interlaced) {
            m_send_data.rtp_interlace_field_indicator = !m_send_data.rtp_interlace_field_indicator;
        }

    }
    m_send_data.rtp_sequence++;
}

size_t RTPVideoMockBufferWriter::build_rtp_header_2110_20_extension(byte_t* buffer)
{
    // build SRD header - 8-14 bytes:
    /* 0                   1                   2                   3
    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |    Extended Sequence Number   |           SRD Length          |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |F|     SRD Row Number          |C|         SRD Offset          |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ */
    auto& video_settings = static_cast<const SMPTE_2110_20_MediaSettings&>(m_media_settings);
    uint16_t extended_sequence_number = htons(static_cast<uint16_t>(m_send_data.rtp_sequence >> 16));
    memcpy(buffer, &extended_sequence_number, sizeof(extended_sequence_number));
    SRDHeader *srd = reinterpret_cast<SRDHeader*>(buffer + RTP_HEADER_EXT_SEQ_NUM_SIZE);
    srd->srd_length = htons(static_cast<uint16_t>(video_settings.raw_packet_payload_size));

    srd->set_srd_row_number(m_send_data.line_number % video_settings.lines_in_frame_field);
    srd->f = m_send_data.rtp_interlace_field_indicator;
    srd->set_srd_offset(m_send_data.srd_offset);
    srd->c = 0;

    return RTP_HEADER_EXT_SEQ_NUM_SIZE + sizeof(SRDHeader);
}

size_t RTPVideoMockBufferWriter::build_rtp_header(byte_t* buffer)
{
    size_t rtp_header_size = build_rtp_header_common(buffer);
    size_t extension_size = build_rtp_header_2110_20_extension(buffer + rtp_header_size);
    return rtp_header_size + extension_size;
}

ReturnStatus RTPVideoBufferWriter::set_next_frame(std::shared_ptr<MediaFrame> frame)
{
    if (frame == nullptr || frame->data == nullptr) {
        std::cerr << "Error: Frame is null or frame data is null" << std::endl;
        return ReturnStatus::failure;
    }
    RTPVideoMockBufferWriter::set_next_frame(frame);
    m_current_frame = std::move(frame);
    m_data_left_in_frame = m_current_frame->data->get_size();
    return ReturnStatus::success;
}

ReturnStatus RTPVideoBufferWriter::write_buffer(void* header_ptr, void* payload_ptr, size_t length_in_strides)
{
    byte_t* header_pointer = reinterpret_cast<byte_t*>(header_ptr);
    byte_t* payload_pointer = reinterpret_cast<byte_t*>(payload_ptr);
    assert(header_pointer);
    assert(payload_ptr);

    if (m_current_frame == nullptr || m_current_frame->data == nullptr ||
        m_current_frame->data->get() == nullptr || !m_payload_mem_utils ||
        m_data_left_in_frame == 0) {
        std::cerr << "Error: Invalid frame state" << std::endl;
        return ReturnStatus::failure;
    }

    // Determine how many complete packets we can process
    size_t packets_to_process = std::min(length_in_strides,
        static_cast<size_t>(m_media_settings.packets_in_frame_field - m_send_data.packet_counter));
    // Limit by available frame data
    size_t max_packets_by_data =
        (m_data_left_in_frame + m_media_settings.raw_packet_payload_size - 1) /
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

    for (size_t i = 0; i < packets_to_process; i++) {
        byte_t* current_packet_pointer = header_pointer + (i * m_media_settings.app_header_stride_size);
        build_rtp_header(current_packet_pointer);
        update_in_frame_state();
    }

    byte_t* frame_ptr = m_current_frame->data->get() + (m_current_frame->data->get_size() - m_data_left_in_frame);
    auto status = m_payload_mem_utils->memory_copy_2D(payload_pointer, m_media_settings.data_stride_size,
        frame_ptr, m_media_settings.raw_packet_payload_size,
        m_media_settings.raw_packet_payload_size, packets_to_process, m_current_frame->data->get_memory_location());

    size_t data_copied = std::min(packets_to_process * m_media_settings.raw_packet_payload_size, m_data_left_in_frame);
    m_data_left_in_frame -= data_copied;

    if (m_data_left_in_frame == 0) {
        m_current_frame = nullptr;
    }

    if (status != ReturnStatus::success) {
        std::cerr << "Failed to 2D copy" << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

size_t RTPVideoBufferWriter::fill_packet(byte_t* buffer)
{
    if (m_current_frame == nullptr || m_current_frame->data == nullptr ||
        m_current_frame->data->get() == nullptr || !m_payload_mem_utils) {
        std::cerr << "Error: Invalid frame state" << std::endl;
        return 0;
    }
    size_t raw_payload_size = m_media_settings.raw_packet_payload_size;
    if (raw_payload_size > m_data_left_in_frame) {
        raw_payload_size = m_data_left_in_frame;
    }

    byte_t* frame_ptr = m_current_frame->data->get() + (m_current_frame->data->get_size() - m_data_left_in_frame);
    m_payload_mem_utils->memory_copy(buffer, frame_ptr, raw_payload_size);
    m_data_left_in_frame -= raw_payload_size;

    // If the frame is fully transmitted, reset the current frame
    if (m_data_left_in_frame == 0) {
        m_current_frame = nullptr;
    }

    return raw_payload_size;
}
