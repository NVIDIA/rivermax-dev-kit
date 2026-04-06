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

#ifndef RDK_SERVICES_ULP_PACKET_WRITERS_RTP_PACKET_WRITER_H_
#define RDK_SERVICES_ULP_PACKET_WRITERS_RTP_PACKET_WRITER_H_

#include "rdk/services/ulp_packet/writers/ulp_packet_writer_interface.h"
#include "rdk/services/ulp_packet/rtp_packet_context.h"

namespace rdk
{
namespace services
{

/**
 * @brief: Base class for RTP media packet writers.
 *
 * This class provides methods to build RTP headers and fill payloads.
 */
class RTPPacketWriter : public IULPPacketWriter {
public:
    /**
     * @brief: Constructor for RTPPacketWriter.
     *
     * Initializes the packet with header and optional payload memory pointers.
     * The payload pointer is optional and used when Header Data Split mode is enabled.
     *
     * @param [in] header_ptr: Pointer to the header memory.
     * @param [in] payload_ptr: Pointer to the payload memory (optional).
     */
    RTPPacketWriter(byte_t* header_ptr, byte_t* payload_ptr):
        IULPPacketWriter(header_ptr, payload_ptr) {}
    virtual ~RTPPacketWriter() = default;
    /**
     * @brief: Fills the RTP packet header.
     *
     * @param [in] context: The packet context containing relevant information.
     * @param [out] size: Reference to store the size of the filled header.
     * @param [in] mem_utils: Memory utilities for header manipulation.
     *
     * @return: The status of the operation.
     */
    ReturnStatus fill_header(const IPacketContext& context, size_t& size, MemoryUtils* mem_utils) override;
    /**
     * @brief: Fills the RTP packet payload.
     *
     * @param [in] context: The packet context containing relevant information.
     * @param [out] size: Reference to store the size of the filled payload.
     * @param [in] mem_utils: Memory utilities for payload manipulation.
     *
     * @return: The status of the operation.
     */
    ReturnStatus fill_payload(const IPacketContext& context, size_t& size, MemoryUtils* mem_utils) override;
    /**
     * @brief: Returns the size of the RTP header.
     *
     * @return: The size of the RTP header in bytes.
     */
    size_t get_header_size() const override;
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_ULP_PACKET_WRITERS_RTP_PACKET_WRITER_H_ */
