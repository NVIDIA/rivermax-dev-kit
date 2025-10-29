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

#ifndef RDK_SERVICES_ULP_PACKET_BUFFER_WR_COMMON_RTP_SMPTE_2110_30_PACKET_H_
#define RDK_SERVICES_ULP_PACKET_BUFFER_WR_COMMON_RTP_SMPTE_2110_30_PACKET_H_

#include "rdk/services/ulp_packet_buffer_wr/common/rtp_packet.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Stub RTP packet structure for SMPTE 2110-30 audio samples.
 *
 * This class provides methods to build RTP headers and fill payloads specific to
 * SMPTE 2110-30 audio samples packets.
 */
class RTP_SMPTE_2110_30_Packet : public RTPPacket
{
public:
    /**
     * @brief: Constructor for RTP_SMPTE_2110_30_Packet.
     *
     * @param [in] header_ptr: Pointer to the header memory.
     * @param [in] payload_ptr: Pointer to the payload memory (optional).
     */
    RTP_SMPTE_2110_30_Packet(byte_t* header_ptr, byte_t* payload_ptr)
        : RTPPacket(header_ptr, payload_ptr) {}
};

/**
 * @brief: Mock RTP packet structure for SMPTE 2110-30 audio samples.
 */
class RTP_SMPTE_2110_30_MockPacket : public RTP_SMPTE_2110_30_Packet
{
public:
    /**
     * @brief: Constructor for RTP_SMPTE_2110_30_MockPacket.
     *
     * Initializes the packet with header and optional payload memory pointers.
     * The payload pointer is optional and used when Header Data Split mode is enabled.
     *
     * @param [in] header_ptr: Pointer to the header memory.
     * @param [in] payload_ptr: Pointer to the payload memory (optional).
     */
    RTP_SMPTE_2110_30_MockPacket(byte_t* header_ptr, byte_t* payload_ptr) :
        RTP_SMPTE_2110_30_Packet(header_ptr, payload_ptr) {}
    /**
     * @brief: Fills the RTP SMPTE 2110-30 packet payload.
     *
     * @param [in] context: The packet context containing relevant information.
     * @param [out] size: Reference to store the size of the filled payload.
     * @param [in] mem_utils: Memory utilities for payload manipulation.
     *
     * @return: The status of the operation.
     */
    ReturnStatus fill_payload(const IPacketContext& context, size_t& size, MemoryUtils* mem_utils) override;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_ULP_PACKET_BUFFER_WR_COMMON_RTP_SMPTE_2110_30_PACKET_H_ */
