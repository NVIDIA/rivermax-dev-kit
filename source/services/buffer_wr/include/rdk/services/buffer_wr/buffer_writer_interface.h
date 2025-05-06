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

#ifndef RDK_SERVICES_BUFFER_WR_BUFFER_WRITER_INTERFACE_H_
#define RDK_SERVICES_BUFFER_WR_BUFFER_WRITER_INTERFACE_H_

#include <cstddef>
#include <memory>

#include "rdk/services/memory_allocation/memory_allocation.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Interface for buffer writers.
 *
 * This class provides an interface for writing data to buffers. It includes methods
 * for writing headers and payloads to buffers. Derived classes must implement the
 * write_buffer method to handle the actual writing process.
 */
class IBufferWriter
{
public:
    /**
     * @brief: Constructor for IBufferWriter.
     *
     * Initializes the buffer writer with memory utilities for header and payload management.
     *
     * @param [in] header_mem_utils: Shared pointer to memory utilities for header management.
     * @param [in] payload_mem_utils: Shared pointer to memory utilities for payload management.
     */
    IBufferWriter(std::shared_ptr<MemoryUtils> header_mem_utils,
        std::shared_ptr<MemoryUtils> payload_mem_utils) :
        m_header_mem_utils(std::move(header_mem_utils)),
        m_payload_mem_utils(std::move(payload_mem_utils)) {}
    /**
     * @brief: Destructor for IBufferWriter.
     */
    virtual ~IBufferWriter() = default;
    /**
     * @brief: Writes data to the buffer when Header Data Split mode is off .
     *
     * This pure virtual method must be implemented by derived classes to handle the
     * writing of headers and payloads to the buffer.
     *
     * @param [in] payload_ptr: Pointer to the payload data.
     * @param [in] buffer_length: Length of the buffer.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus write_buffer(void* payload_ptr, size_t buffer_length) = 0;
    /**
     * @brief: Writes data to the buffer when Header Data Split is on.
     *
     * This pure virtual method must be implemented by derived classes to handle the
     * writing of headers and payloads to the buffer.
     *
     * @param [in] header_ptr: Pointer to the header data.
     * @param [in] payload_ptr: Pointer to the payload data.
     * @param [in] buffer_length: Length of the buffer.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus write_buffer(void* header_ptr, void* payload_ptr, size_t buffer_length) = 0;
protected:
    /* Memory utilities for header management. */
    std::shared_ptr<MemoryUtils> m_header_mem_utils;
    /* Memory utilities for payload management. */
    std::shared_ptr<MemoryUtils> m_payload_mem_utils;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_BUFFER_WR_BUFFER_WRITER_INTERFACE_H_ */
