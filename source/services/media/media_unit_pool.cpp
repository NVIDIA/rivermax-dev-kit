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

#include "rdk/services/media/media_unit_pool.h"

using namespace rdk::services;

MediaUnitPool::MediaUnitPool(size_t media_unit_count, size_t media_unit_size, SMPTEStandard smpte_standard,
    MemoryAllocator& mem_allocator) :
    m_media_unit_size(media_unit_size),
    m_media_unit_count(media_unit_count),
    m_memory_location(mem_allocator.get_memory_location()),
    m_smpte_standard(smpte_standard)
{
    m_total_memory_size = media_unit_count * media_unit_size;
    m_total_memory_size = mem_allocator.align_length(m_total_memory_size);
    m_memory_block = static_cast<byte_t*>(mem_allocator.allocate_aligned(
        m_total_memory_size, mem_allocator.get_page_size()));

    if (!m_memory_block) {
        throw std::runtime_error("Failed to allocate memory for MediaUnitPool");
    }

    m_all_media_units.reserve(media_unit_count);
    for (size_t i = 0; i < media_unit_count; ++i) {
        byte_t* media_unit_memory = m_memory_block + (i * media_unit_size);
        m_all_media_units.emplace_back(media_unit_memory, media_unit_size, m_smpte_standard, m_memory_location);
        m_available_indices.push(i);
    }
}

MediaUnitPool::MediaUnitPool(size_t media_unit_count, size_t media_unit_size, SMPTEStandard smpte_standard,
    byte_t* memory_block, size_t memory_size, MemoryLocation memory_location) :
    m_memory_block(memory_block),
    m_media_unit_size(media_unit_size),
    m_media_unit_count(media_unit_count),
    m_total_memory_size(media_unit_count * media_unit_size),
    m_memory_location(memory_location),
    m_smpte_standard(smpte_standard)
{
    if (!m_memory_block) {
        throw std::runtime_error("No memory provided for for MediaUnitPool");
    }

    if (memory_size < m_total_memory_size) {
        throw std::invalid_argument("Insufficient external memory provided for MediaUnitPool");
    }

    // Initialize units and indices
    m_all_media_units.reserve(media_unit_count);
    for (size_t index = 0; index < media_unit_count; ++index) {
        byte_t* unit_memory = memory_block + (index * media_unit_size);
        m_all_media_units.emplace_back(unit_memory, media_unit_size, m_smpte_standard, m_memory_location);
        m_available_indices.push(index);
    }
}

MediaUnitPool::~MediaUnitPool()
{
    m_in_destruction = true;
    m_available_indices = std::queue<size_t>();
    m_all_media_units.clear();
}

void MediaUnitPool::return_media_unit_to_pool(size_t index)
{
    if (m_in_destruction || m_stop) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    m_available_indices.push(index);
    m_cv.notify_one();
}

std::shared_ptr<MediaUnit> MediaUnitPool::get_media_unit()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_available_indices.empty() || m_stop) {
        return nullptr;
    }

    size_t index = m_available_indices.front();
    m_available_indices.pop();

    return std::shared_ptr<MediaUnit>(
        &m_all_media_units[index],
        [this, index](MediaUnit*) { this->return_media_unit_to_pool(index); }
    );
}

std::shared_ptr<MediaUnit> MediaUnitPool::get_media_unit_blocking()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [this] { return !m_available_indices.empty() || m_stop; });

    if (m_stop) {
        return nullptr;
    }

    size_t index = m_available_indices.front();
    m_available_indices.pop();

    return std::shared_ptr<MediaUnit>(
        &m_all_media_units[index],
        [this, index](MediaUnit*) { this->return_media_unit_to_pool(index); }
    );
}

size_t MediaUnitPool::get_available_media_units_count() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_available_indices.size();
}

void MediaUnitPool::stop()
{
    m_stop = true;
    m_cv.notify_all();
}
