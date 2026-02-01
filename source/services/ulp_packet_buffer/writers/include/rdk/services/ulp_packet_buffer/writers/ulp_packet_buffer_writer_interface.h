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

#ifndef RDK_SERVICES_ULP_PACKET_BUFFER_WRITERS_ULP_PACKET_BUFFER_WRITER_INTERFACE_H_
#define RDK_SERVICES_ULP_PACKET_BUFFER_WRITERS_ULP_PACKET_BUFFER_WRITER_INTERFACE_H_

#include "rdk/services/memory_allocation/memory_allocation.h"
#include "rdk/services/media/media.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Interface for Upper Layer Protocol (ULP) packet buffer writers.
 *
 * This class provides an interface for writing packets data to memory.
 * It includes methods for writing packet headers and payloads to allocated buffer memory,
 * supporting both Header Data Split and non-Header Data Split modes.
 * Derived classes must implement the @ref write_buffer methods to handle the actual packet
 * buffer writing process.
 */
class IULPPacketBufferWriter
{
public:
    /**
     * @brief: Constructor for IULPPacketBufferWriter.
     *
     * Initializes the packet buffer writer with memory utilities for header and payload
     * management.
     *
     * @param [in] header_mem_utils: Shared pointer to memory utilities for header management.
     * @param [in] payload_mem_utils: Shared pointer to memory utilities for payload management.
     */
    IULPPacketBufferWriter(std::shared_ptr<MemoryUtils> header_mem_utils,
        std::shared_ptr<MemoryUtils> payload_mem_utils) :
        m_header_mem_utils(std::move(header_mem_utils)),
        m_payload_mem_utils(std::move(payload_mem_utils)) {}
    /**
     * @brief: Destructor for IULPPacketBufferWriter.
     */
    virtual ~IULPPacketBufferWriter() = default;
    /**
     * @brief: Writes packet buffer when Header Data Split mode is off.
     *
     * This pure virtual method must be implemented by derived classes to handle the
     * writing of complete packets (header + payload) to a single contiguous buffer.
     *
     * @param [in] payload_ptr: Pointer to the buffer where complete packet data
     *                          (header + payload) will be written.
     * @param [in] buffer_length: Length of the buffer in strides.
     * @param [in] payload_sizes: Optional array to fill with actual payload sizes if not nullptr.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus write_buffer(void* payload_ptr, size_t buffer_length, uint16_t* payload_sizes = nullptr) = 0;
    /**
     * @brief: Writes packet buffer when Header Data Split mode is on.
     *
     * This pure virtual method must be implemented by derived classes to handle the
     * writing of packet headers and payloads to separate buffers.
     *
     * @param [in] header_ptr: Pointer to the header buffer.
     * @param [in] payload_ptr: Pointer to the payload buffer.
     * @param [in] buffer_length: Length of the buffer in strides.
     * @param [in] header_sizes: Optional array to fill with actual header sizes if not nullptr.
     * @param [in] payload_sizes: Optional array to fill with actual payload sizes if not nullptr.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus write_buffer(void* header_ptr, void* payload_ptr, size_t buffer_length,
                                      uint16_t* header_sizes = nullptr, uint16_t* payload_sizes = nullptr) = 0;
    /**
     * @brief: Sets the next media unit to be processed.
     *
     * @param [in] media_unit: Pointer to the media unit.
     *
     * @return: Return status of the operation.
     */
    virtual ReturnStatus set_next_media_unit(std::shared_ptr<MediaUnit> media_unit) = 0;
    /**
     * @brief: Sets the start time for packet timestamp generation.
     *
     * @param [in] time_ns: The start time.
     */
    virtual void set_start_time(uint64_t time_ns) = 0;
    /**
     * @brief: Returns the number of packets needed for the next chunk.
     *
     * @return: Number of packets for the next chunk.
     */
    virtual size_t get_num_packets_for_next_chunk() const = 0;

protected:
    /* Memory utilities for header management. */
    std::shared_ptr<MemoryUtils> m_header_mem_utils;
    /* Memory utilities for payload management. */
    std::shared_ptr<MemoryUtils> m_payload_mem_utils;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_ULP_PACKET_BUFFER_WRITERS_ULP_PACKET_BUFFER_WRITER_INTERFACE_H_ */
