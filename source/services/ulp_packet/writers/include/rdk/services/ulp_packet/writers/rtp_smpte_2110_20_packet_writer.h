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

#ifndef RDK_SERVICES_ULP_PACKET_WRITERS_RTP_SMPTE_2110_20_PACKET_WRITER_H_
#define RDK_SERVICES_ULP_PACKET_WRITERS_RTP_SMPTE_2110_20_PACKET_WRITER_H_

#include "rdk/services/ulp_packet/writers/rtp_packet_writer.h"
#include "rdk/services/ulp_packet/rtp_smpte_2110_20_packet_context.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: RTP packet writer for SMPTE 2110-20 video data.
 *
 * This class provides methods to build RTP headers and fill payloads specific to
 * SMPTE 2110-20 video data packets.
 *
 * @note: Currently supports single SRD header per packet.
 */
class RTP_SMPTE_2110_20_PacketWriter : public RTPPacketWriter
{
public:
    /**
     * @brief: Constructor for RTP_SMPTE_2110_20_PacketWriter.
     *
     * Initializes the packet with header and optional payload memory pointers.
     * The payload pointer is optional and used when Header Data Split mode is enabled.
     *
     * @param [in] header_ptr: Pointer to the header memory.
     * @param [in] payload_ptr: Pointer to the payload memory (optional).
     */
    RTP_SMPTE_2110_20_PacketWriter(byte_t* header_ptr, byte_t* payload_ptr);
    virtual ~RTP_SMPTE_2110_20_PacketWriter() = default;
    /**
     * @brief: Fills the RTP SMPTE 2110-20 packet header.
     *
     * @param [in] context: The packet context containing relevant information.
     * @param [out] size: Reference to store the size of the filled header.
     * @param [in] mem_utils: Memory utilities for header manipulation.
     *
     * @return: The status of the operation.
     */
    ReturnStatus fill_header(const IPacketContext& context, size_t& size, MemoryUtils* mem_utils) override;
    /**
     * @brief: Returns the size of the RTP SMPTE 2110-20 packet header.
     *
     * @return: The size of the RTP SMPTE 2110-20 packet header in bytes.
     */
    size_t get_header_size() const override;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_ULP_PACKET_WRITERS_RTP_SMPTE_2110_20_PACKET_WRITER_H_ */
