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

#ifndef RDK_SERVICES_ULP_PACKET_WRITERS_RTP_SMPTE_2110_40_PACKET_WRITER_H_
#define RDK_SERVICES_ULP_PACKET_WRITERS_RTP_SMPTE_2110_40_PACKET_WRITER_H_

#include "rdk/services/ulp_packet/writers/rtp_packet_writer.h"
#include "rdk/services/ulp_packet/rtp_smpte_2110_40_packet_context.h"
#include "rdk/services/media/ancillary_metadata.h"

namespace rdk
{
namespace services
{

/**
 * @brief: Helper class to write ancillary data payloads into a buffer.
 *
 * Provides method to write ancillary data packets that are carried in the RTP payload
 * based on section 2.1 of RFC 8331 - RTP Payload for SMPTE ST 291-1 Ancillary Data.
 */
class AncillaryDataPacketWriter
{
public:
    AncillaryDataPacketWriter() = delete;
    /**
     * @brief: Writes ancillary data packet into the buffer.
     *
     * @param [out] buffer: Pointer to the buffer to write the packet into.
     * @param [in] user_data_bytes: Pointer to array of 8-bit user data words (before parity bits are added).
     * @param [in] ancillary_data_descriptor: Ancillary data descriptor containing header and user data offset.
     *
     * @return: Total number of bytes written to the buffer.
     */
    static size_t write_ancillary_data(byte_t* buffer, byte_t* user_data_bytes,
                                       const AncillaryDataDescriptor& ancillary_data_descriptor);
};

/**
 * @brief: RTP packet writer for SMPTE 2110-40 ancillary data.
 *
 * This class provides methods to build RTP headers and fill payloads specific to
 * SMPTE 2110-40 ancillary data packets.
 */
class RTP_SMPTE_2110_40_PacketWriter : public RTPPacketWriter
{
public:
    /**
     * @brief: Constructor for RTP_SMPTE_2110_40_PacketWriter.
     *
     * Initializes the packet with header and optional payload memory pointers.
     * The payload pointer is optional and used when Header Data Split mode is enabled.
     *
     * @param [in] header_ptr: Pointer to the header memory.
     * @param [in] payload_ptr: Pointer to the payload memory (optional).
     */
    RTP_SMPTE_2110_40_PacketWriter(byte_t* header_ptr, byte_t* payload_ptr);
    virtual ~RTP_SMPTE_2110_40_PacketWriter() = default;
    /**
     * @brief: Fills the RTP SMPTE 2110-40 packet header.
     *
     * @param [in] context: The packet context containing relevant information.
     * @param [out] size: Reference to store the size of the filled header.
     * @param [in] mem_utils: Memory utilities for header manipulation.
     *
     * @return: The status of the operation.
     */
    ReturnStatus fill_header(const IPacketContext& context, size_t& size, MemoryUtils* mem_utils) override;
    /**
     * @brief: Fills the RTP SMPTE 2110-40 packet payload.
     *
     * @param [in] context: The packet context containing relevant information.
     * @param [out] size: Reference to store the size of the filled payload.
     * @param [in] mem_utils: Memory utilities for payload manipulation.
     *
     * @return: The status of the operation.
     */
    ReturnStatus fill_payload(const IPacketContext& context, size_t& size, MemoryUtils* mem_utils) override;
    /**
     * @brief: Returns the size of the RTP SMPTE 2110-40 packet header.
     *
     * @return: The size of the RTP SMPTE 2110-40 packet header in bytes.
     */
    size_t get_header_size() const override;
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_ULP_PACKET_WRITERS_RTP_SMPTE_2110_40_PACKET_WRITER_H_ */
