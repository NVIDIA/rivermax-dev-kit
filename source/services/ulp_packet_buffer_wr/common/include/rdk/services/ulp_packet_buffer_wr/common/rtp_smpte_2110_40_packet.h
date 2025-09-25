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

#ifndef RDK_SERVICES_ULP_PACKET_BUFFER_WR_COMMON_RTP_SMPTE_2110_40_PACKET_H_
#define RDK_SERVICES_ULP_PACKET_BUFFER_WR_COMMON_RTP_SMPTE_2110_40_PACKET_H_

#include "rdk/services/ulp_packet_buffer_wr/common/rtp_packet.h"
#include "rdk/services/media/media.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Context for RTP SMPTE 2110-40 packets.
 *
 * This struct extends the @ref RTPPacketContext to include fields
 * specific to SMPTE 2110-40 ancillary data packets based on section 2.1 of
 * RFC 8331 - RTP Payload for Society of Motion Picture and Television Engineers (SMPTE)
 * ST 291-1 Ancillary Data.
 */
struct RTP_SMPTE_2110_40_PacketContext : public RTPPacketContext
{
    uint32_t extended_sequence_number = 0;   /**< 32-bit extended sequence number */
    uint16_t length;                         /**< Number of octets of the ANC data RTP payload */
    uint32_t ancillary_count = 1;            /**< Number of ancillary data packets */
    uint8_t field_indicator = 0;             /**< Field indicator specifying RTP timestamp
                                                  relation to video fields */
    bool c_flag = false;                     /**< Color channel flag */
    uint16_t line_number = 0;                /**< Line number of the ANC data */
    uint16_t horizontal_offset = 0;          /**< Horizontal offset of the ANC data */
    bool s_flag = false;                     /**< Data stream flag */
    uint8_t stream_number = 0;               /**< Data stream number */
    uint8_t did = 0;                         /**< Data identification word */
    uint8_t sdid = 0;                        /**< Secondary data identification word */
    std::vector<uint8_t> user_data;          /**< User data */
    size_t user_data_size_bytes = 16;        /**< User data size in bytes */
};

/**
 * @brief: RTP packet structure for SMPTE 2110-40 ancillary data.
 *
 * This class provides methods to build RTP headers and fill payloads specific to
 * SMPTE 2110-40 ancillary data packets.
 */
class RTP_SMPTE_2110_40_Packet : public RTPPacket
{
public:
    /**
     * @brief: Constructor for RTP_SMPTE_2110_40_Packet.
     *
     * Initializes the packet with header and optional payload memory pointers.
     * The payload pointer is optional and used when Header Data Split mode is enabled.
     *
     * @param [in] header_ptr: Pointer to the header memory.
     * @param [in] payload_ptr: Pointer to the payload memory (optional).
     */
    RTP_SMPTE_2110_40_Packet(byte_t* header_ptr, byte_t* payload_ptr);
    virtual ~RTP_SMPTE_2110_40_Packet() = default;
    /**
     * @brief: Fills the RTP SMPTE 2110-40 packet header.
     *
     * @param [in] context: The packet context containing relevant information.
     * @param [out] size: Reference to store the size of the filled header.
     * @param [in] mem_utils: Memory utilities for header manipulation.
     *
     * @return: The status of the operation.
     */
    ReturnStatus fill_header(const PacketContext& context, size_t& size, MemoryUtils* mem_utils) override;
    /**
     * @brief: Fills the RTP SMPTE 2110-40 packet payload.
     *
     * @param [in] context: The packet context containing relevant information.
     * @param [out] size: Reference to store the size of the filled payload.
     * @param [in] mem_utils: Memory utilities for payload manipulation.
     *
     * @return: The status of the operation.
     */
    ReturnStatus fill_payload(const PacketContext& context, size_t& size, MemoryUtils* mem_utils) override;
    /**
     * @brief: Returns the size of the RTP SMPTE 2110-40 packet header.
     *
     * @return: The size of the RTP SMPTE 2110-40 packet header in bytes.
     */
    size_t get_header_size() const override;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_ULP_PACKET_BUFFER_WR_COMMON_RTP_SMPTE_2110_40_PACKET_H_ */
