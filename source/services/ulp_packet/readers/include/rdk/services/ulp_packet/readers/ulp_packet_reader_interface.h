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

#ifndef RDK_SERVICES_ULP_PACKET_READERS_ULP_PACKET_READER_INTERFACE_H_
#define RDK_SERVICES_ULP_PACKET_READERS_ULP_PACKET_READER_INTERFACE_H_

#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/utils/defs.h"
#include "rdk/services/ulp_packet/ulp_packet_context.h"

namespace rdk
{
namespace services
{

/**
 * @brief: Interface for Upper Layer Protocol (ULP) packet readers.
 *
 * This class provides an interface for parsing packet headers and accessing payloads.
 * Derived classes must implement the @ref parse_header and @ref parse_payload
 * methods to handle the actual packet parsing process.
 */
class IULPPacketReader
{
protected:
    const byte_t* m_header_ptr = nullptr;
    const byte_t* m_payload_ptr = nullptr;

public:
    /**
     * @brief: Constructor for IULPPacketReader.
     *
     * Initializes the reader with header and optional payload memory pointers.
     * The payload pointer is optional and used when Header Data Split mode is enabled.
     *
     * @param [in] header_ptr: Const pointer to the header memory.
     * @param [in] payload_ptr: Const pointer to the payload memory (optional, for HDS mode).
     */
    IULPPacketReader(const byte_t* header_ptr, const byte_t* payload_ptr = nullptr) :
        m_header_ptr(header_ptr),
        m_payload_ptr(payload_ptr) {}
    /**
     * @brief: Destructor for IULPPacketReader.
     */
    virtual ~IULPPacketReader() = default;
    /**
     * @brief: Sets the packet pointers for header and payload.
     *
     * @param [in] header_ptr: Const pointer to the header memory.
     * @param [in] payload_ptr: Const pointer to the payload memory (optional).
     */
    void set_packet(const byte_t* header_ptr, const byte_t* payload_ptr = nullptr)
    {
        m_header_ptr = header_ptr;
        m_payload_ptr = payload_ptr;
    }
    /**
     * @brief: Parses the packet header into context.
     *
     * @param [out] context: The packet context to populate with parsed header fields.
     *
     * @return: Status of the operation.
     */
     virtual ReturnStatus parse_header(IPacketContext& context) = 0;
    /**
     * @brief: Parses the packet payload.
     *
     * @param [out] context: The packet context to populate with parsed payload data.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus parse_payload(IPacketContext& context) { return ReturnStatus::success; }
    /**
     * @brief: Returns const pointer to payload data.
     *
     * @return: Const pointer to the payload start.
     */
    virtual const byte_t* get_payload_ptr() { return m_payload_ptr; }
    /**
     * @brief: Returns the size of the payload.
     *
     * @return: Payload size in bytes.
     */
    virtual size_t get_payload_size() const = 0;
    /**
     * @brief: Returns the size of the packet header.
     *
     * @return: Header size in bytes.
     */
    virtual size_t get_header_size() const = 0;
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_ULP_PACKET_READERS_ULP_PACKET_READER_INTERFACE_H_ */
