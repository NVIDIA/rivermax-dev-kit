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

#include "rdk/services/ulp_packet/writers/rtp_packet_writer.h"

using namespace rivermax::dev_kit::services;

/**
 * @brief: Standard RTP packet header
 *
 * Based on RFC 3550 - RTP: A Transport Protocol for Real-Time Applications.
 */
struct RTPHeader {
    uint8_t cc : 4;            /**< CSRC count */
    uint8_t extension : 1;     /**< Extension bit */
    uint8_t padding : 1;       /**< Padding bit */
    uint8_t version : 2;       /**< RTP version */
    uint8_t payload_type : 7;  /**< Payload type */
    uint8_t marker : 1;        /**< Marker bit */
    uint16_t sequence_number;  /**< Sequence number */
    uint32_t timestamp;        /**< Timestamp */
    uint32_t ssrc;             /**< Synchronization source (SSRC) identifier */
};

ReturnStatus RTPPacketWriter::fill_header(const IPacketContext& context, size_t& size, MemoryUtils* mem_utils)
{
    const auto& rtp_packet_context = static_cast<const RTPPacketContext&>(context);

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

    RTPHeader* p_rtp_header = reinterpret_cast<RTPHeader*>(m_header_ptr );
    p_rtp_header->version = rtp_packet_context.version;
    p_rtp_header->padding = rtp_packet_context.padding;
    p_rtp_header->extension = rtp_packet_context.extension;
    p_rtp_header->cc = rtp_packet_context.cc;
    p_rtp_header->payload_type = rtp_packet_context.payload_type;
    p_rtp_header->sequence_number = htons(rtp_packet_context.sequence);
    p_rtp_header->timestamp = htonl(static_cast<uint32_t>(rtp_packet_context.timestamp.integer())); // Truncate per ST 2110-10
    p_rtp_header->ssrc = htonl(rtp_packet_context.ssrc);
    p_rtp_header->marker = rtp_packet_context.marker;

    size = sizeof(RTPHeader);
    return ReturnStatus::success;
}

ReturnStatus RTPPacketWriter::fill_payload(const IPacketContext& context, size_t& size, MemoryUtils* mem_utils)
{
    const auto& rtp_packet_context = static_cast<const RTPPacketContext&>(context);

    size = rtp_packet_context.payload_size;
    if (!rtp_packet_context.payload_ptr) {
        return ReturnStatus::success;
    }

    if (m_payload_ptr == nullptr) {
        m_payload_ptr = m_header_ptr + get_header_size();
    }
    mem_utils->memory_copy(m_payload_ptr, rtp_packet_context.payload_ptr, size);
    return ReturnStatus::success;
}

size_t RTPPacketWriter::get_header_size() const
{
    return sizeof(RTPHeader);
}
