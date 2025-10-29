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

#ifndef RDK_SERVICES_ULP_PACKET_BUFFER_WR_COMMON_RTP_PACKET_H_
#define RDK_SERVICES_ULP_PACKET_BUFFER_WR_COMMON_RTP_PACKET_H_

#include "rdk/services/ulp_packet_buffer_wr/common/ulp_packet_interface.h"
#include "rdk/services/media/media.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Context for RTP packets.
 *
 * This struct extends @ref PacketContext with RTP-specific fields based on Section 5.1
 * of RFC 3550 - RTP: A Transport Protocol for Real-Time Applications.
 */
struct RTPPacketContext : public PacketContext
{
    uint8_t version = 2;                 /**< RTP version, currently 2 */
    bool padding = false;                /**< Padding bit */
    bool extension = false;              /**< Extension bit */
    uint8_t cc = 0;                      /**< CSRC count */
    uint8_t payload_type = 0;            /**< Payload type */
    bool marker = false;                 /**< Marker bit */
    uint16_t sequence = 0;               /**< 16-bit RTP sequence number */
    uint32_t ssrc = 0;                   /**< Synchronization source (SSRC) identifier */

    uint32_t counter = 0;                /**< Packet counter */
    size_t payload_size = 0;             /**< Size of the RTP payload */
    std::shared_ptr<MediaUnit> current_media_unit = nullptr;
    size_t data_left_in_media_unit_in_bytes = 0;
};

/**
 * @brief: Base class for RTP media packets.
 *
 * This class provides methods to build RTP headers and fill payloads.
 */
class RTPPacket : public IULPPacket {
public:
    /**
     * @brief: Constructor for RTPPacket.
     *
     * Initializes the packet with header and optional payload memory pointers.
     * The payload pointer is optional and used when Header Data Split mode is enabled.
     *
     * @param [in] header_ptr: Pointer to the header memory.
     * @param [in] payload_ptr: Pointer to the payload memory (optional).
     */
    RTPPacket(byte_t* header_ptr, byte_t* payload_ptr): 
        IULPPacket(header_ptr, payload_ptr) {}
    virtual ~RTPPacket() = default;
    /**
     * @brief: Fills the RTP packet header.
     *
     * @param [in] context: The packet context containing relevant information.
     * @param [out] size: Reference to store the size of the filled header.
     * @param [in] mem_utils: Memory utilities for header manipulation.
     *
     * @return: The status of the operation.
     */
    ReturnStatus fill_header(const PacketContext& context, size_t& size, MemoryUtils* mem_utils) override;
    /**
     * @brief: Fills the RTP packet payload.
     *
     * @param [in] context: The packet context containing relevant information.
     * @param [out] size: Reference to store the size of the filled payload.
     * @param [in] mem_utils: Memory utilities for payload manipulation.
     *
     * @return: The status of the operation.
     */
    ReturnStatus fill_payload(const PacketContext& context, size_t& size, MemoryUtils* mem_utils) override;
    /**
     * @brief: Returns the size of the RTP header.
     *
     * @return: The size of the RTP header in bytes.
     */
    size_t get_header_size() const override;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_ULP_PACKET_BUFFER_WR_COMMON_RTP_PACKET_H_ */
