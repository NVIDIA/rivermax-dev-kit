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
#include <cstring>

#include "rdk/services/buffer_wr/rtp_video_buffer_writer.h"
#include "rdk/services/media/media.h"

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
    size_t app_header_stride_size, size_t data_stride_size, uint16_t packet_payload_size,
    std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils) :
    RTPMediaBufferWriter(media_settings, app_header_stride_size, data_stride_size,
        packet_payload_size, std::move(header_mem_utils), std::move(payload_mem_utils))
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
    m_send_data.srd_offset = (m_send_data.srd_offset + m_media_settings.pixels_per_packet) %
        (m_media_settings.resolution.width);
    if (!((m_send_data.packet_counter + 1) % m_media_settings.packets_in_line)) {
        // Prepare line number for next iteration:
        m_send_data.line_number = (m_send_data.line_number + 1) % m_media_settings.lines_in_frame_field;
    }
    if (++m_send_data.packet_counter == m_media_settings.packets_in_frame_field) {
        // ST2210-20: the timestamp SHOULD be the same for each packet of the frame/field.
        m_send_data.rtp_timestamp += static_cast<uint32_t>(m_media_settings.ticks_per_frame);
        m_send_data.packet_counter = 0;
        if (m_media_settings.video_scan_type == VideoScanType::Interlaced) {
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

    uint16_t extended_sequence_number = htons(static_cast<uint16_t>(m_send_data.rtp_sequence >> 16));
    memcpy(buffer, &extended_sequence_number, sizeof(extended_sequence_number));
    SRDHeader *srd = reinterpret_cast<SRDHeader*>(buffer + RTP_HEADER_EXT_SEQ_NUM_SIZE);
    srd->srd_length = htons(static_cast<uint16_t>(m_media_settings.raw_packet_payload_size));

    srd->set_srd_row_number(m_send_data.line_number % m_media_settings.lines_in_frame_field);
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
    RTPVideoMockBufferWriter::set_next_frame(frame);
    m_current_frame = std::move(frame);
    m_data_left_in_frame = m_current_frame->data.get_size();
    return ReturnStatus::success;
}

size_t RTPVideoBufferWriter::fill_packet(byte_t* buffer)
{
    if (m_current_frame == nullptr || !m_payload_mem_utils) {
        return 0;
    }
    auto raw_payload_size = m_media_settings.raw_packet_payload_size;
    if (raw_payload_size > m_data_left_in_frame) {
        raw_payload_size = m_data_left_in_frame;
    }

    byte_t* frame_ptr = m_current_frame->data.get() + (m_current_frame->data.get_size() - m_data_left_in_frame);
    m_payload_mem_utils->memory_copy(buffer, frame_ptr, raw_payload_size);
    m_data_left_in_frame -= raw_payload_size;

    // If the frame is fully transmitted, reset the current frame
    if (m_data_left_in_frame == 0) {
        m_current_frame = nullptr;
    }

    return raw_payload_size;
}
