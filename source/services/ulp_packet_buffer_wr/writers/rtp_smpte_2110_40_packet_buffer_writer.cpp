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

/**
 * @brief: ST 2110-40 Ancillary RTP extension header.
 *
 * RTP header extension for ancillary data streams following the standard
 * RTP header. Provides extended sequence numbering, payload length, and
 * field indicator for interlaced formats.
 *
 */
struct AncillaryRTPExtension
{
    /**
     * @brief: Sets the field indicator value.
     *
     * @param [in] f: Field indicator (0=progressive, 1=field1, 2=field2).
     */
    void set_field_indicator(uint8_t f)
    {
        reserved_byte1 = (f & 0x03) << 6;
    }

    uint16_t extended_sequence_number;
    uint16_t length;
    uint8_t anc_count;
    uint8_t reserved_byte1;
    uint16_t reserved_bytes2_3;
};

static constexpr size_t RTP_HEADER_EXT_SEQ_NUM_SIZE = 2;

RTP_SMPTE_2110_40_PacketBufferWriter::RTP_SMPTE_2110_40_PacketBufferWriter(const MediaSettings& media_settings,
    std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils) :
    RTPMediaPacketBufferWriter(media_settings, std::move(header_mem_utils), std::move(payload_mem_utils)),
    m_field_indicator(0)
{
    set_stream_properties();
}

ReturnStatus RTP_SMPTE_2110_40_PacketBufferWriter::set_next_media_unit(std::shared_ptr<MediaUnit> media_unit)
{
    reset_in_media_unit_state();
    return ReturnStatus::success;
}

void RTP_SMPTE_2110_40_PacketBufferWriter::reset_in_media_unit_state()
{
    m_send_data.packet_counter = 0;
    m_field_indicator = 0;
}

void RTP_SMPTE_2110_40_PacketBufferWriter::update_in_media_unit_state()
{
    // ST 2110-40: timestamp is the SAME for all packets in a media unit (like video)
    // Only increment after the last packet of the media unit
    if (++m_send_data.packet_counter >= m_media_settings.packets_in_media_unit) {
        // Timestamp changes every media unit (90kHz clock)
        m_send_data.rtp_timestamp += static_cast<uint32_t>(m_media_settings.ticks_per_media_unit);
        m_send_data.packet_counter = 0;

        // Toggle field indicator for interlaced content
        // Note: This is simplified; real implementation should check video_scan_type
        m_field_indicator = (m_field_indicator == 0) ? 1 : 0;
    }
    m_send_data.rtp_sequence++;
}

size_t RTP_SMPTE_2110_40_PacketBufferWriter::build_rtp_header_2110_40_extension(byte_t* buffer)
{
    // Build ST 2110-40 ancillary extension header
    /* 0                   1                   2                   3
       0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
       |    Extended Sequence Number   |           Length              |
       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
       | ANC_Count     |F|   reserved                                  |
       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ */

    // Extended sequence number (upper 16 bits of 48-bit sequence)
    uint16_t extended_seq_num = htons(static_cast<uint16_t>(m_send_data.rtp_sequence >> 16));
    memcpy(buffer, &extended_seq_num, sizeof(extended_seq_num));

    // Length field (payload length, excluding RTP headers)
    uint16_t length = htons(static_cast<uint16_t>(m_media_settings.raw_packet_payload_size));
    memcpy(buffer + 2, &length, sizeof(length));

    // ANC_Count: simplified to 1 for now (number of ancillary data packets)
    buffer[4] = 1;

    // F field (2 bits) + reserved (22 bits)
    buffer[5] = (m_field_indicator & 0x03) << 6;  // F is top 2 bits
    buffer[6] = 0;  // Reserved
    buffer[7] = 0;  // Reserved

    return 8;  // Size of ST 2110-40 extension header
}

size_t RTP_SMPTE_2110_40_PacketBufferWriter::build_rtp_header(byte_t* buffer)
{
    // Build standard RTP header (marker bit set correctly by base class)
    // Marker = 1 only on last packet of frame (packets_in_media_unit - 1)
    size_t rtp_header_size = build_rtp_header_common(buffer);

    // Build ST 2110-40 extension header
    size_t extension_size = build_rtp_header_2110_40_extension(buffer + rtp_header_size);

    return rtp_header_size + extension_size;
}

size_t RTP_SMPTE_2110_40_PacketBufferWriter::fill_packet(byte_t* buffer)
{
    // Mock implementation: no actual payload filling
    // Real implementation would copy ancillary data here
    return m_media_settings.raw_packet_payload_size;
}
