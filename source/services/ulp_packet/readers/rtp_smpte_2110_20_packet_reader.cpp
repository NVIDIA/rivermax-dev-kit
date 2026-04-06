/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "rdk/services/ulp_packet/readers/rtp_smpte_2110_20_packet_reader.h"
#include "rdk/services/ulp_packet/network_byte_order.h"

#include <cassert>

using namespace rdk::services;

ReturnStatus RTP_SMPTE_2110_20_PacketReader::parse_header(IPacketContext& context)
{
    assert(m_header_ptr);

    ReturnStatus status = RTPPacketReader::parse_header(context);

    auto& rtp_packet_context = static_cast<RTP_SMPTE_2110_20_PacketContext&>(context);

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

    const SRDHeader* srd_header_ptr = reinterpret_cast<const SRDHeader*>(m_header_ptr + RTPPacketReader::get_header_size());

    rtp_packet_context.extended_sequence_number =
        (static_cast<uint32_t>(ntohs(srd_header_ptr->sequence_number)) << 16) | rtp_packet_context.sequence;
    rtp_packet_context.srd_length = ntohs(srd_header_ptr->srd_length);
    rtp_packet_context.line_number = srd_header_ptr->get_srd_line_number();
    rtp_packet_context.srd_offset = srd_header_ptr->get_srd_offset();
    rtp_packet_context.rtp_interlace_field_indicator = srd_header_ptr->f;

    m_payload_size = rtp_packet_context.srd_length;

    return status;
}
