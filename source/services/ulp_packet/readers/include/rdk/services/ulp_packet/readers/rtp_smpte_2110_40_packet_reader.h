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

#ifndef RDK_SERVICES_ULP_PACKET_READERS_RTP_SMPTE_2110_40_PACKET_READER_H_
#define RDK_SERVICES_ULP_PACKET_READERS_RTP_SMPTE_2110_40_PACKET_READER_H_

#include <cstdint>

#include "rdk/services/ulp_packet/readers/rtp_packet_reader.h"
#include "rdk/services/ulp_packet/rtp_smpte_2110_40_extension.h"
#include "rdk/services/ulp_packet/rtp_smpte_2110_40_packet_context.h"
#include "rdk/services/media/ancillary_metadata.h"

namespace rdk
{
namespace services
{

/**
 * @brief: Decodes SMPTE ST 2110-40 ancillary data sub-packets from packed 10-bit word streams.
 *
 * Sub-packet wire layout (RFC 8331 section 2.1 / SMPTE ST 291-1):
 *
 * 0                   1                   2                   3
 * 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |C|      Line_Number      |    Horizontal_Offset  |S| Stream_Num|
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |         DID       |        SDID       |     Data_Count    |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *                          User_Data_Words...
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *             |   Checksum_Word   |         word_align            |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *
 * Each 10-bit word: bit[9]=NOT(parity), bit[8]=even_parity(bits[7:0]), bits[7:0]=data
 */
class AncillaryDataPacketReader
{
public:
    AncillaryDataPacketReader() = delete;
    /**
     * @brief: Decodes one ANC data sub-packet (fixed header + 10-bit word region) from buffer.
     *
     * On any failure output_ptr may have been partially written.
     *
     * @param [in] buffer: Start of the ANC sub-packet on the wire.
     * @param [in] remaining_bytes: Bytes available.
     * @param [in] current_output_offset: Stored as ancillary_data_descriptor.user_data_offset.
     * @param [out] ancillary_data_descriptor: Receives the decoded ANC descriptor.
     * @param [out] output_ptr: Destination for parity-stripped UDW bytes.
     *
     * @return: Bytes consumed (sub-packet wire size), or 0 on failure.
     */
    static size_t read_ancillary_data(const byte_t* buffer, size_t remaining_bytes,
        size_t current_output_offset,
        AncillaryDataDescriptor& ancillary_data_descriptor,
        byte_t* output_ptr);
};

/**
 * @brief: RTP packet reader for SMPTE ST 2110-40 ancillary data streams (RFC 8331).
 *
 * Parses the RTP header, the ST 2110-40 extension header, and all ANC sub-packets in the payload.
 */
class RTP_SMPTE_2110_40_PacketReader : public RTPPacketReader
{
public:
    /**
     * @brief: Constructs a reader bound to the given packet memory region.
     *
     * @param [in] header_ptr: Pointer to the first byte of the RTP header.
     * @param [in] payload_ptr: Optional separate payload pointer (HDS mode); when null, payload follows the header inline.
     */
    RTP_SMPTE_2110_40_PacketReader(const byte_t* header_ptr, const byte_t* payload_ptr = nullptr)
        : RTPPacketReader(header_ptr, payload_ptr) {}
    virtual ~RTP_SMPTE_2110_40_PacketReader() = default;
    /**
     * @brief: Parses the RTP base header and the ST 2110-40 RTP extension header.
     *
     * @param [out] context: Must be an @ref RTP_SMPTE_2110_40_PacketContext.
     *
     * @return: ReturnStatus::success, or ReturnStatus::failure on parse error.
     */
    ReturnStatus parse_header(IPacketContext& context) override;
    /**
     * @brief: Decodes all ANC data sub-packets from the RTP payload.
     *
     * @pre parse_header() must be called successfully on the same context before this method.
     *
     * Each decoded sub-packet appends one @ref AncillaryDataDescriptor to context.descriptors
     * and writes its UDW bytes to context.payload_ptr. Each appended descriptor's user_data_offset
     * is relative to context.payload_ptr at the time of this call. On failure, descriptors and
     * UDW bytes from sub-packets decoded before the error are retained.
     *
     * @param [in,out] context: Must be an @ref RTP_SMPTE_2110_40_PacketContext with descriptors
     *                          and payload_ptr set by the caller.
     *
     * @return: ReturnStatus::success on full decode, ReturnStatus::failure on truncation,
     *          parity error, or checksum mismatch.
     */
    ReturnStatus parse_payload(IPacketContext& context) override;
    /**
     * @brief: Returns the total header size.
     *
     * @return: RTP header size + ancillary RTP extension size.
     */
    inline size_t get_header_size() const override
    {
        return RTPPacketReader::get_header_size() + sizeof(AncillaryRTPExtension);
    }
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_ULP_PACKET_READERS_RTP_SMPTE_2110_40_PACKET_READER_H_ */
