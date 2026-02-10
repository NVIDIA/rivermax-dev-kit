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

#include <cstring>

#include "rdk/services/ulp_packet/writers/rtp_smpte_2110_20_packet_writer.h"

using namespace rivermax::dev_kit::services;

/**
 * @brief: ST 2110-20 SRD RTP extension header.
 *
 * RTP header extension for ST 2110-20 Sample Row Data (SRD) packets
 * based on section 4.2 of RFC 4175 - RTP Payload Format for Uncompressed Video.
 */
struct SRDHeader {
    uint16_t sequence_number;                /**< Extended Sequence Number: 16 bits */
    uint16_t srd_length;                     /**< SRD Length: 16 bits */

    uint8_t srd_line_number_8_to_14_7bit: 7; /**< SRD line number: 15 bits */
    uint8_t f: 1;                            /**< Identifies which field the scan line belongs to,
                                                  for interlaced data : 1 bit */
    uint8_t srd_line_number_0_to_7_8bit;     /**< SRD line number: 15 bits */

    uint8_t srd_offset_8_to_14_7bit: 7;      /**< SRD offset: 15 bits */
    uint8_t c: 1;                            /**< Field identification: 1 bit */
    uint8_t srd_offset_0_to_7_8bit;          /**< SRD offset: 15 bits */

    void set_srd_line_number(uint16_t srd_line_number) {
        srd_line_number_0_to_7_8bit = static_cast<uint8_t>(srd_line_number);
        srd_line_number_8_to_14_7bit = static_cast<uint8_t>(srd_line_number >> 8);
    }

    void set_srd_offset(uint16_t srd_offset) {
        srd_offset_0_to_7_8bit = static_cast<uint8_t>(srd_offset);
        srd_offset_8_to_14_7bit = static_cast<uint8_t>(srd_offset >> 8);
    }
};

RTP_SMPTE_2110_20_PacketWriter::RTP_SMPTE_2110_20_PacketWriter(byte_t* header_ptr, byte_t* payload_ptr)
    : RTPPacketWriter(header_ptr, payload_ptr)
{
}

ReturnStatus RTP_SMPTE_2110_20_PacketWriter::fill_header(const IPacketContext& context, size_t& size, MemoryUtils* mem_utils)
{
    const auto& rtp_packet_context = static_cast<const RTP_SMPTE_2110_20_PacketContext&>(context);

    ReturnStatus status = RTPPacketWriter::fill_header(context, size, mem_utils);

    /**
     * @brief: ST 2110-20 SRD RTP extension header.
     *
     * Using extended RTP format based on RFC 4175 - RTP Payload Format for Uncompressed
     * Video.
     *
     * 0                   1                   2                   3
     * 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
     * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     * |    Extended Sequence Number   |           SRD Length          |
     * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     * |F|     SRD Row Number          |C|         SRD Offset          |
     * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     */

    SRDHeader* p_srd_header = reinterpret_cast<SRDHeader*>(m_header_ptr  + size);
    uint16_t high_bits = (rtp_packet_context.extended_sequence_number >> 16) & 0xFFFF;
    p_srd_header->sequence_number = htons(high_bits);

    p_srd_header->srd_length = htons(rtp_packet_context.srd_length);

    p_srd_header->set_srd_line_number(rtp_packet_context.line_number);
    p_srd_header->f = rtp_packet_context.rtp_interlace_field_indicator;
    p_srd_header->set_srd_offset(rtp_packet_context.srd_offset);
    p_srd_header->c = 0;

    size = size + sizeof(SRDHeader);
    return status;
}

size_t RTP_SMPTE_2110_20_PacketWriter::get_header_size() const
{
    return RTPPacketWriter::get_header_size() + sizeof(SRDHeader);
}
