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

#ifndef RDK_CORE_MEMORY_LAYOUT_STREAM_MEMORY_LAYOUT_H_
#define RDK_CORE_MEMORY_LAYOUT_STREAM_MEMORY_LAYOUT_H_

#include <cstddef>
#include <vector>

#include <rivermax_api.h>

#include "rdk/services/error_handling/return_status.h"
#include "rdk/core/memory_layout/memory_layout_interface.h"

namespace rdk
{
namespace core
{

/**
 * @brief: Stream buffer memory sizes.
 *
 * This struct holds the memory sizes for a stream header, payload and auxiliary buffers.
 */
struct StreamBufferMemorySizes
{
    size_t header_buffer_size = 0;
    size_t payload_buffer_size = 0;
    size_t auxiliary_buffer_size = 0;
};

/**
 * @brief: Header-Payload memory layout configuration for memory components.
 *
 * This struct holds the memory layout configuration details for a memory component, including pointers
 * to header and payload buffers, memory keys, and memory sizes.
 */
struct StreamMemoryLayout
{
    /**
     * @brief: Constructor for StreamMemoryLayout.
     *
     * @param [in] register_memory_: Flag indicating whether memory should be registered.
     * @param [in] num_mem_keys: Number of memory keys to allocate.
     */
    StreamMemoryLayout(bool register_memory_ = false, size_t num_mem_keys = 0) :
        register_memory(register_memory_) {
        if (register_memory) {
            header_memory_keys.resize(num_mem_keys);
            payload_memory_keys.resize(num_mem_keys);
        }
    }
    /**
     * @brief: Destructor for StreamMemoryLayout.
     */
    ~StreamMemoryLayout() = default;
    /* Pointer to the header memory buffer. */
    void* header_memory_ptr = nullptr;
    /* Pointer to the payload memory buffer. */
    void* payload_memory_ptr = nullptr;
    /* Pointer to the auxiliary memory buffer. */
    void* auxiliary_memory_ptr = nullptr;
    /* Size of the header memory buffer. */
    size_t header_memory_size = 0;
    /* Size of the payload memory buffer. */
    size_t payload_memory_size = 0;
    /* Size of the auxiliary memory buffer. */
    size_t auxiliary_memory_size = 0;
    /* Flag indicating whether memory should be registered. */
    bool register_memory = false;
    /* Memory keys for header memory. */
    std::vector<rmx_mkey_id> header_memory_keys;
    /* Memory keys for payload memory. */
    std::vector<rmx_mkey_id> payload_memory_keys;
};

/**
 * @brief: Header-Payload memory layout response for memory components.
 *
 * This struct holds the single memory layout configuration response for a memory component.
 */
struct StreamMemoryLayoutResponse : public MemoryLayoutResponse
{
    /**
     * @brief: Constructor for StreamMemoryLayoutResponse.
     *
     * @param [in] register_memory_: Flag indicating whether memory should be registered.
     * @param [in] num_mem_keys: Number of memory keys to allocate.
     */
    StreamMemoryLayoutResponse(bool register_memory_ = false, size_t num_mem_keys = 0) :
        memory_layout(register_memory_, num_mem_keys) {}
    /**
     * @brief: Destructor for StreamMemoryLayoutResponse.
     */
    virtual ~StreamMemoryLayoutResponse() = default;
    /* Header-Payload memory layout configuration for memory components. */
    StreamMemoryLayout memory_layout;
};

/**
 * @brief: Memory layout request for memory components.
 *
 * This struct holds the single memory layout configuration request from a memory component.
 */
struct StreamMemoryLayoutRequest : public MemoryLayoutRequest
{
    /**
     * @brief: Destructor for StreamMemoryLayoutRequest.
     */
    virtual ~StreamMemoryLayoutRequest() = default;
    /* Required memory buffer sizes. */
    StreamBufferMemorySizes buffer_sizes;
};

/**
 * @brief: Interface for Header-Payload memory layout components.
 *
 * This interface defines the methods that memory layout components must implement.
 */
using IStreamMemoryLayoutComponent = IMemoryLayoutComponent<StreamMemoryLayoutRequest, StreamMemoryLayoutResponse>;

} // namespace core
} // namespace rdk

#endif /* RDK_CORE_MEMORY_LAYOUT_STREAM_MEMORY_LAYOUT_H_ */
