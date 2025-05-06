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

#include <string>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

#include <rivermax_api.h>
#include "rt_threads.h"

#include "rdk/apps/rmax_base_app.h"
#include "rdk/apps/rmax_base_memory_strategy.h"
#include "rdk/io_node/io_node.h"

using namespace rivermax::dev_kit::apps;
using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

RmaxBaseMemoryStrategy::RmaxBaseMemoryStrategy(
    MemoryAllocator& header_allocator, MemoryAllocator& payload_allocator,
    IONodeMemoryUtils& memory_utils,
    std::vector<rmx_device_iface> device_interfaces,
    size_t num_of_mem_regions,
    bool app_allocated_memory,
    bool register_memory) :
    m_header_allocator(header_allocator),
    m_payload_allocator(payload_allocator),
    m_memory_utils(memory_utils),
    m_device_interfaces(std::move(device_interfaces)),
    m_num_of_mem_regions(num_of_mem_regions),
    m_app_allocated_memory(app_allocated_memory),
    m_register_memory(register_memory)
{
}

ReturnStatus RmaxBaseMemoryStrategy::determine_memory_layout()
{
    if (m_memory_layout_determined) {
        return ReturnStatus::success;
    }

    m_header_memory_size = 0;
    m_payload_memory_size = 0;
    m_header_payload_subcomponents_buffer_sizes.clear();
    m_header_payload_subcomponents_buffer_sizes.reserve(m_memory_subcomponents.size());

    for (const auto& subcomponent : m_memory_subcomponents) {
        ReturnStatus status = subcomponent->initialize_memory_layout();
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to initialize memory layout for memory subcomponent" << std::endl;
            return status;
        }
    }

    for (const auto& subcomponent : m_memory_subcomponents) {
        HeaderPayloadMemoryLayoutRequest subcomponent_memory_layout;
        ReturnStatus status = subcomponent->determine_memory_layout(subcomponent_memory_layout);
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to determine memory layout for memory subcomponent" << std::endl;
            return status;
        }

        m_header_memory_size += subcomponent_memory_layout.header_payload_buffers_size.first;
        m_payload_memory_size += subcomponent_memory_layout.header_payload_buffers_size.second;
        m_header_payload_subcomponents_buffer_sizes.push_back(
            subcomponent_memory_layout.header_payload_buffers_size);
    }
    m_memory_layout_determined = true;
    std::cout << "Application requires " << m_header_memory_size << " bytes of header memory and "
              << m_payload_memory_size << " bytes of payload memory" << std::endl;

    return ReturnStatus::success;
}

ReturnStatus RmaxBaseMemoryStrategy::allocate_memory()
{
    if (!m_app_allocated_memory || m_memory_allocated) {
        return ReturnStatus::success;
    }
    if (!m_memory_layout_determined || m_payload_memory_size == 0) {
        std::cerr << "Memory layout not determined" << std::endl;
        return ReturnStatus::failure;
    }

    bool alloc_successful = allocate_aligned(m_header_memory_size, m_payload_memory_size,
                                             m_header_buffer, m_payload_buffer);
    if (alloc_successful) {
        std::cout << "Allocated " << m_header_memory_size << " bytes for header"
            << " at address " << static_cast<void*>(m_header_buffer)
            << " and " <<  m_payload_memory_size << " bytes for payload"
            << " at address " << static_cast<void*>(m_payload_buffer) << std::endl;
    } else {
        std::cerr << "Failed to allocate memory" << std::endl;
        return ReturnStatus::failure;
    }
    m_memory_allocated = true;
    return ReturnStatus::success;
}

ReturnStatus RmaxBaseMemoryStrategy::register_memory()
{
    if (!m_app_allocated_memory || !m_register_memory || m_memory_registered) {
        return ReturnStatus::success;
    }
    if (!m_memory_allocated || m_payload_buffer == nullptr) {
        std::cerr << "Memory was not allocated" << std::endl;
        return ReturnStatus::failure;
    }
    m_header_mem_regions.resize(m_num_of_mem_regions);
    m_payload_mem_regions.resize(m_num_of_mem_regions);

    auto register_memory = [&](auto& mem_region, auto buffer, auto memory_size) {
        for (size_t i = 0; i < m_num_of_mem_regions; ++i) {
            mem_region[i].addr = buffer;
            mem_region[i].length = memory_size;
            mem_region[i].mkey = RMX_MKEY_INVALID;
            if (memory_size) {
                rmx_mem_reg_params mem_registry;
                rmx_init_mem_registry(&mem_registry, &m_device_interfaces[i]);
                rmx_status status = rmx_register_memory(&mem_region[i], &mem_registry);
                if (status != RMX_OK) {
                    std::cerr << "Failed to register memory " << " with status: " << status << std::endl;
                    return ReturnStatus::failure;
                }
            }
        }
        return ReturnStatus::success;
    };
    ReturnStatus status = register_memory(m_header_mem_regions, m_header_buffer, m_header_memory_size);
    if (status != ReturnStatus::success) {
        return status;
    }

    status = register_memory(m_payload_mem_regions, m_payload_buffer, m_payload_memory_size);
    if (status != ReturnStatus::success) {
        return status;
    }
    m_memory_registered = true;
    return ReturnStatus::success;
}

void RmaxBaseMemoryStrategy::unregister_memory()
{
    if (!m_app_allocated_memory || !m_register_memory || !m_memory_registered) {
        return;
    }

    auto unregister_memory = [&](auto& mem_region) {
        for (size_t i = 0; i < mem_region.size(); ++i) {
            rmx_status status = rmx_deregister_memory(&mem_region[i], &m_device_interfaces[i]);
            if (status != RMX_OK) {
                std::cerr << "Failed to deregister memory " << " with status: " << status << std::endl;
            }
        }
    };

    if (m_header_buffer) {
        unregister_memory(m_header_mem_regions);
    }
    unregister_memory(m_payload_mem_regions);
    m_memory_registered = false;
}

ReturnStatus RmaxBaseMemoryStrategy::apply_memory_layout()
{
    byte_t* header_ptr = m_header_buffer;
    byte_t* payload_ptr = m_payload_buffer;
    ReturnStatus status;

    for (size_t i = 0; i < m_memory_subcomponents.size(); ++i) {
        auto& subcomponent = m_memory_subcomponents[i];
        auto& subcomponent_header_payload_buffers_size = m_header_payload_subcomponents_buffer_sizes[i];

        if (m_app_allocated_memory) {
            status = apply_memory_layout_helper(subcomponent_header_payload_buffers_size,
                    header_ptr, payload_ptr, *subcomponent);
            if (header_ptr) {
                header_ptr += subcomponent_header_payload_buffers_size.first;
            }
            payload_ptr += subcomponent_header_payload_buffers_size.second;
        } else {
            status = apply_memory_layout_helper({0, 0}, nullptr, nullptr, *subcomponent);
        }
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to apply memory layout for component." << std::endl;
            return status;
        }
    }

    return ReturnStatus::success;
}

ReturnStatus RmaxBaseMemoryStrategy::apply_memory_layout_helper(
    const std::pair<size_t, size_t>& component_header_payload_buffers_size,
    byte_t* header_ptr, byte_t* payload_ptr,
    IHeaderPayloadMemoryLayoutComponent& memory_component)
{
    HeaderPayloadMemoryLayoutResponse component_memory_layout_response;
    auto& component_memory_layout = component_memory_layout_response.memory_layout;
    component_memory_layout.header_memory_ptr = header_ptr;
    component_memory_layout.payload_memory_ptr = payload_ptr;
    component_memory_layout.header_memory_size = component_header_payload_buffers_size.first;
    component_memory_layout.payload_memory_size = component_header_payload_buffers_size.second;
    component_memory_layout.register_memory = m_register_memory;

    if (component_memory_layout.register_memory) {
        component_memory_layout.header_memory_keys.resize(m_num_of_mem_regions);
        component_memory_layout.payload_memory_keys.resize(m_num_of_mem_regions);
        for (int key_idx = 0; key_idx < m_num_of_mem_regions; ++key_idx) {
            component_memory_layout.header_memory_keys[key_idx] = m_header_mem_regions[key_idx].mkey;
            component_memory_layout.payload_memory_keys[key_idx] = m_payload_mem_regions[key_idx].mkey;
        }
    }

    auto status = memory_component.apply_memory_layout(component_memory_layout_response);
    if (status != ReturnStatus::success) {
        std::cerr << "Failed to set memory layout for component." << std::endl;
        return status;
    }
    return ReturnStatus::success;
}

bool RmaxBaseMemoryStrategy::allocate_aligned(
    size_t header_size, size_t payload_size, byte_t*& header_ptr, byte_t*& payload_ptr)
{
    header_ptr = payload_ptr = nullptr;
    if (header_size) {
        header_ptr = static_cast<byte_t*>(m_header_allocator.allocate_aligned(header_size,
            m_header_allocator.get_page_size()));
    }
    payload_ptr = static_cast<byte_t*>(m_payload_allocator.allocate_aligned(payload_size,
        m_payload_allocator.get_page_size()));
    return payload_ptr && (header_size == 0 || header_ptr);
}

ReturnStatus RmaxBaseMemoryStrategy::add_memory_subcomponent(std::shared_ptr<IHeaderPayloadMemoryLayoutComponent> component) {
    m_memory_subcomponents.push_back(std::move(component));
    return ReturnStatus::success;
}
