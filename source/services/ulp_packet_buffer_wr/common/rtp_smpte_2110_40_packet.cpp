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

#include "rdk/services/ulp_packet_buffer_wr/common/rtp_smpte_2110_40_packet.h"

using namespace rivermax::dev_kit::services;

/**
 * @brief: ST 2110-40 Ancillary RTP extension header.
 *
 * RTP header extension for ancillary data streams following the standard
 * RTP header based on section 2.1 of RFC 8331 - RTP Payload for Society of
 * Motion Picture and Television Engineers (SMPTE) ST 291-1 Ancillary Data.
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

RTP_SMPTE_2110_40_Packet::RTP_SMPTE_2110_40_Packet(byte_t* header_ptr, byte_t* payload_ptr)
    : RTPPacket(header_ptr, payload_ptr)
{
}

ReturnStatus RTP_SMPTE_2110_40_Packet::fill_header(const PacketContext& context, size_t& size, MemoryUtils* mem_utils)
{
    const auto& rtp_packet_context = static_cast<const RTP_SMPTE_2110_40_PacketContext&>(context);

    ReturnStatus status = RTPPacket::fill_header(context, size, mem_utils);

    /**
     * @brief: ST 2110-40 Ancillary RTP Header Extension Format
     *
     * Using extended RTP format based on RFC 8331 - RTP Payload for Society of 
     * Motion Picture and Television Engineers (SMPTE) ST 291-1 Ancillary Data.
     *
     * 0                   1                   2                   3
     * 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
     * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     * |    Extended Sequence Number   |           Length              |
     * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     * | ANC_Count     |F|   reserved                                  |
     * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ 
     */

    AncillaryRTPExtension* p_ancillary_header = reinterpret_cast<AncillaryRTPExtension*>(m_header_ptr  + size);
    uint16_t high_bits = (rtp_packet_context.extended_sequence_number >> 16) & 0xFFFF;
    p_ancillary_header->extended_sequence_number = htons(high_bits);

    p_ancillary_header->length = htons(rtp_packet_context.length);
    p_ancillary_header->anc_count = rtp_packet_context.ancillary_count;
    p_ancillary_header->set_field_indicator(rtp_packet_context.field_indicator);

    size = size + sizeof(AncillaryRTPExtension);
    return status;
}

ReturnStatus RTP_SMPTE_2110_40_Packet::fill_payload(const PacketContext& context, size_t& size, MemoryUtils* mem_utils)
{
    const auto& rtp_packet_context = static_cast<const RTP_SMPTE_2110_40_PacketContext&>(context);
    // Mock implementation: no actual payload filling
    // TODO : Implement actual payload filling based on ancillary data structure
    /**
     *  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     *  |C|   Line_Number=9     |   Horizontal_Offset   |S| StreamNum=0 |
     *  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     *  |         DID       |        SDID       |  Data_Count=0x84  |
     *  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     *                           User_Data_Words...
     *  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     *              |   Checksum_Word   |         word_align            |
     *  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ 
    */
    size = rtp_packet_context.payload_size;
    return ReturnStatus::success;
}

size_t RTP_SMPTE_2110_40_Packet::get_header_size() const
{
    return RTPPacket::get_header_size() + sizeof(AncillaryRTPExtension);
}
