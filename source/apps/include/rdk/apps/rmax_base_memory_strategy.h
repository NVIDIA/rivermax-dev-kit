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

#ifndef RDK_APPS_RMAX_BASE_MEMORY_STRATEGY_H_
#define RDK_APPS_RMAX_BASE_MEMORY_STRATEGY_H_

#include <vector>
#include <memory>
#include <climits>

#include <rivermax_api.h>
#include "CLI/CLI.hpp"

#include "rdk/apps/rmax_base_app.h"
#include "rdk/apps/memory_strategy_interface.h"
#include "rdk/io_node/io_node.h"
#include "rdk/core/memory_layout/header_payload_memory_layout.h"

using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
{
namespace apps
{

/**
 * @brief: Base class for Rivermax apps memory strategy implementation.
 *
 * This class provides a base implementation for managing memory layout,
 * allocation, registration, and application for memory components.
 */
class RmaxBaseMemoryStrategy : public IMemoryStrategy
{
protected:
    /* Header memory allocator */
    MemoryAllocator& m_header_allocator;
    /* Payload memory allocator */
    MemoryAllocator& m_payload_allocator;
    /* Memory utilities */
    IONodeMemoryUtils& m_memory_utils;
    /* Flag indicating if memory layout is determined */
    bool m_memory_layout_determined = false;
    /* Flag indicating if memory is allocated */
    bool m_memory_allocated = false;
    /* Flag indicating if memory is registered */
    bool m_memory_registered = false;
    /* Memory Layout components objects container */
    std::vector<std::shared_ptr<IHeaderPayloadMemoryLayoutComponent>> m_memory_subcomponents;
    /* Size of header memory buffer */
    size_t m_header_memory_size = 0;
    /* Size of payload memory buffer */
    size_t m_payload_memory_size = 0;
    /* Buffer for packet header (if header-data split is enabled) */
    byte_t* m_header_buffer = nullptr;
    /* Buffer for packet payload */
    byte_t* m_payload_buffer = nullptr;
    /* Subcomponents requested buffer sizes for header and payload */
    std::vector<std::pair<size_t, size_t>> m_header_payload_subcomponents_buffer_sizes;
    /* NIC device interfaces */
    std::vector<rmx_device_iface> m_device_interfaces;
    /* Memory regions for header memory allocated for each device interface */
    std::vector<rmx_mem_region> m_header_mem_regions;
    /* Memory regions for payload memory allocated for each device interface */
    std::vector<rmx_mem_region> m_payload_mem_regions;
    /* Number of memory keys stream */
    size_t m_num_of_mem_regions = 0;
    /* Flag indicating if the application allocated memory */
    bool m_app_allocated_memory = false;
    /* Flag indicating if memory should be registered */
    bool m_register_memory = false;
public:
    /**
     * @brief: Constructor for RmaxBaseMemoryStrategy.
     *
     * @param [in] header_allocator: Reference to the header memory allocator.
     * @param [in] payload_allocator: Reference to the payload memory allocator.
     * @param [in] memory_utils: Reference to the IONodeMemoryUtils object.
     * @param [in] device_interfaces: Vector of NIC device interfaces.
     * @param [in] num_of_mem_regions: Number of memory regions per stream.
     * @param [in] app_allocated_memory: Flag indicating if the application allocated memory.
     * @param [in] register_memory: Flag indicating if memory should be registered.
     */
    RmaxBaseMemoryStrategy(MemoryAllocator& header_allocator, MemoryAllocator& payload_allocator,
        IONodeMemoryUtils& memory_utils,
        std::vector<rmx_device_iface> device_interfaces,
        size_t num_of_mem_regions,
        bool app_allocated_memory = false,
        bool register_memory = false);
    /**
     * @brief: Destructor for RmaxBaseMemoryStrategy.
     */
    virtual ~RmaxBaseMemoryStrategy() = default;

    ReturnStatus determine_memory_layout() override;
    ReturnStatus allocate_memory() override;
    ReturnStatus register_memory() override;
    void unregister_memory() override;
    ReturnStatus apply_memory_layout() override;
    /**
     * @brief: Adds a memory component.
     *
     * @param [in] component: Shared pointer to the memory layout component.
     *
     * @return: Returns status of the operation.
     */
    virtual ReturnStatus add_memory_subcomponent(
        std::shared_ptr<IHeaderPayloadMemoryLayoutComponent> component);
protected:
    /**
     * @brief: Allocates memory and aligns it to page size.
     *
     * @param [in] header_size: Requested header memory size.
     * @param [in] payload_size: Requested payload memory size.
     * @param [out] header_ptr: Allocated header memory pointer.
     * @param [out] payload_ptr: Allocated payload memory pointer.
     *
     * @return: True if successful.
     */
    virtual bool allocate_aligned(size_t header_size, size_t payload_size,
        byte_t*& header_ptr, byte_t*& payload_ptr);
    /**
     * @brief: Sets the memory pointers and sizes for a single memory layout component.
     *
     * @param [in] component_header_payload_buffers_size: The sizes of the header and payload buffer.
     * @param [in] header_ptr: Pointer to the header buffer.
     * @param [in] payload_ptr: Pointer to the payload buffer.
     * @param [in] memory_component: The memory subcomponent to apply memory layout for.
     *
     * @return: Returns status of the operation.
     */
    virtual ReturnStatus apply_memory_layout_helper(
        const std::pair<size_t, size_t>& component_header_payload_buffers_size,
        byte_t* header_ptr, byte_t* payload_ptr,
        IHeaderPayloadMemoryLayoutComponent& memory_component);
};

} // namespace apps
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_APPS_RMAX_BASE_MEMORY_STRATEGY_H_ */
