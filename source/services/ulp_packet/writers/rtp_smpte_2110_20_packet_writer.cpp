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

#include <cstring>

#include "rdk/services/ulp_packet/writers/rtp_smpte_2110_20_packet_writer.h"
#include "rdk/services/ulp_packet/network_byte_order.h"
#include "rdk/services/ulp_packet/rtp_smpte_2110_20_extension.h"

using namespace rdk::services;

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
