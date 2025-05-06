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

#include <cstddef>
#include <memory>
#include <vector>
#include <utility>

#include "rdk/io_node/common/io_node_memory_utils.h"
#include "rdk/services/memory_allocation/memory_allocator_interface.h"

using namespace rivermax::dev_kit::io_node;

IONodeMemoryUtils::IONodeMemoryUtils(MemoryAllocator& header_allocator, MemoryAllocator& payload_allocator) :
    m_header_allocator(header_allocator),
    m_payload_allocator(payload_allocator)
{
}

std::pair<size_t, size_t> IONodeMemoryUtils::align_buffer_sizes(const std::pair<size_t, size_t>& buffer_sizes) const
{
    return { m_header_allocator.align_length(buffer_sizes.first), m_payload_allocator.align_length(buffer_sizes.second) };
}

std::vector<std::pair<size_t, size_t>> IONodeMemoryUtils::build_aligned_buffer_sizes(const std::vector<std::pair<size_t, size_t>>& buffer_sizes) const
{
    std::vector<std::pair<size_t, size_t>> aligned_buffer_sizes;
    for (const auto& buffer_size : buffer_sizes) {
        size_t header_buffer_size = m_header_allocator.align_length(buffer_size.first);
        size_t payload_buffer_size = m_payload_allocator.align_length(buffer_size.second);
        aligned_buffer_sizes.push_back(std::make_pair(header_buffer_size, payload_buffer_size));
    }
    return aligned_buffer_sizes;
}

std::pair<size_t, size_t> IONodeMemoryUtils::get_total_memory_requirements(const std::vector<std::pair<size_t, size_t>>& buffer_sizes) const
{
    size_t header_memory_size = 0;
    size_t payload_memory_size = 0;

    for (const auto& buffer_size : buffer_sizes) {
        header_memory_size += buffer_size.first;
        payload_memory_size += buffer_size.second;
    }

    return {header_memory_size, payload_memory_size};
}

std::shared_ptr<MemoryUtils> IONodeMemoryUtils::get_header_memory_utils() const
{
    return m_header_allocator.get_memory_utils();
}

std::shared_ptr<MemoryUtils> IONodeMemoryUtils::get_payload_memory_utils() const
{
    return m_payload_allocator.get_memory_utils();
}

size_t IONodeMemoryUtils::get_aligned_header_size(size_t size) const
{
    return m_header_allocator.align_length(size);
}

size_t IONodeMemoryUtils::get_aligned_payload_size(size_t size) const
{
    return m_payload_allocator.align_length(size);
}
