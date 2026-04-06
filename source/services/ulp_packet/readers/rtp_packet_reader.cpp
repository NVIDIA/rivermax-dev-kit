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

#include "rdk/services/ulp_packet/readers/rtp_packet_reader.h"
#include "rdk/services/ulp_packet/network_byte_order.h"
#include "rdk/services/ulp_packet/rtp_header.h"

#include <cassert>

using namespace rdk::services;

ReturnStatus RTPPacketReader::parse_header(IPacketContext& context)
{
    assert(m_header_ptr);

    auto& rtp_packet_context = static_cast<RTPPacketContext&>(context);

    /**
     * @brief: Standard RTP header format.
     *
     * Using RTP format based on RFC 3550 - RTP: A Transport Protocol for Real-Time
     * Applications.
     *
     * 0                   1                   2                   3
     * 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
     * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     * | V |P|X|  CC   |M|     PT      |            SEQ                |
     * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     * |                           timestamp                           |
     * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     * |                           ssrc                                |
     * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     */

    const RTPHeader* p_rtp_header = reinterpret_cast<const RTPHeader*>(m_header_ptr);
    rtp_packet_context.version = p_rtp_header->version;
    rtp_packet_context.padding = p_rtp_header->padding;
    rtp_packet_context.extension = p_rtp_header->extension;
    rtp_packet_context.cc = p_rtp_header->cc;
    rtp_packet_context.payload_type = p_rtp_header->payload_type;
    rtp_packet_context.marker = (p_rtp_header->marker != 0);
    rtp_packet_context.sequence = ntohs(p_rtp_header->sequence_number);
    rtp_packet_context.timestamp = ntohl(p_rtp_header->timestamp);
    rtp_packet_context.ssrc = ntohl(p_rtp_header->ssrc);

    return ReturnStatus::success;
}
