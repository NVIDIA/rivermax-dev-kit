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

#ifndef RDK_SERVICES_ULP_PACKET_WRITERS_ULP_PACKET_WRITER_INTERFACE_H_
#define RDK_SERVICES_ULP_PACKET_WRITERS_ULP_PACKET_WRITER_INTERFACE_H_

#include <memory>

#include "rdk/services/utils/defs.h"
#include "rdk/services/memory_allocation/memory_allocation.h"
#include "rdk/services/ulp_packet/ulp_packet_context.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Interface for Upper Layer Protocol (ULP) packet writers.
 *
 * This class provides an interface for building packet headers and payloads.
 * Derived classes must implement the @ref fill_header and @ref fill_payload
 * methods to handle the actual packet construction process.
 */
class IULPPacketWriter
{
protected:
    byte_t* m_header_ptr  = nullptr;
    byte_t* m_payload_ptr  = nullptr;

public:
    /**
     * @brief: Constructor for IULPPacketWriter.
     *
     * Initializes the packet with header and optional payload memory pointers.
     * The payload pointer is optional and used when Header Data Split mode is enabled.
     *
     * @param [in] header_ptr: Pointer to the header memory.
     * @param [in] payload_ptr: Pointer to the payload memory (optional).
     */
    IULPPacketWriter(byte_t* header_ptr, byte_t* payload_ptr = nullptr) :
        m_header_ptr (header_ptr), m_payload_ptr (payload_ptr) {}
    /**
     * @brief: Destructor for IULPPacketWriter.
     */
    virtual ~IULPPacketWriter() = default;
    /**
     * @brief: Sets the packet pointers for header and payload.
     *
     * @param [in] header_ptr: Pointer to the header memory.
     * @param [in] payload_ptr: Pointer to the payload memory (optional).
     */
    void set_packet(byte_t* header_ptr, byte_t* payload_ptr = nullptr)
    {
        m_header_ptr = header_ptr;
        m_payload_ptr = payload_ptr;
    }
    /**
     * @brief: Fills the packet header.
     *
     * @param [in] context: The packet context containing relevant information.
     * @param [out] size: Reference to store the size of the filled header.
     * @param [in] mem_utils: Memory utilities for header manipulation.
     *
     * @return: The status of the operation.
     */
    virtual ReturnStatus fill_header(const IPacketContext& context, size_t& size, MemoryUtils* mem_utils = nullptr) = 0;
    /**
     * @brief: Fills the packet payload.
     *
     * @param [in] context: The packet context containing relevant information.
     * @param [out] size: Reference to store the size of the filled payload.
     * @param [in] mem_utils: Memory utilities for payload manipulation.
     *
     * @return: The status of the operation.
     */
    virtual ReturnStatus fill_payload(const IPacketContext& context, size_t& size, MemoryUtils* mem_utils = nullptr) = 0;
    /**
     * @brief: Returns the size of the packet header.
     *
     * @return: The size of the packet header in bytes.
     */
    virtual size_t get_header_size() const = 0;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_ULP_PACKET_WRITERS_ULP_PACKET_WRITER_INTERFACE_H_ */
