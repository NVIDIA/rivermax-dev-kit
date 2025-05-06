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

#include "rdk/services/media/media_frame_pool.h"

using namespace rivermax::dev_kit::services;

MediaFramePool::MediaFramePool(size_t frame_count, size_t frame_size, MemoryAllocator& mem_allocator) :
    m_frame_size(frame_size),
    m_frame_count(frame_count)
{
    m_total_memory_size = frame_count * frame_size;
    m_total_memory_size = mem_allocator.align_length(m_total_memory_size);
    m_memory_block = static_cast<byte_t*>(mem_allocator.allocate_aligned(
        m_total_memory_size, mem_allocator.get_page_size()));

    if (!m_memory_block) {
        throw std::runtime_error("Failed to allocate memory for MediaFramePool");
    }

    m_all_frames.reserve(frame_count);
    for (size_t i = 0; i < frame_count; ++i) {
        byte_t* frame_memory = m_memory_block + (i * frame_size);
        m_all_frames.emplace_back(frame_memory, frame_size);
        m_available_indices.push(i);
    }
}

MediaFramePool::MediaFramePool(size_t frame_count, size_t frame_size, byte_t* memory_block, size_t memory_size ) :
    m_memory_block(memory_block),
    m_frame_size(frame_size),
    m_frame_count(frame_count),
    m_total_memory_size(frame_count * frame_size)
{
    if (!m_memory_block) {
        throw std::runtime_error("No memory provided for for MediaFramePool");
    }

    if (memory_size < m_total_memory_size) {
        throw std::invalid_argument("Insufficient external memory provided for MediaFramePool");
    }

    // Initialize frames and indices
    m_all_frames.reserve(frame_count);
    for (size_t index = 0; index < frame_count; ++index) {
        byte_t* frame_memory = memory_block + (index * frame_size);
        m_all_frames.emplace_back(frame_memory, frame_size);
        m_available_indices.push(index);
    }
}

MediaFramePool::~MediaFramePool()
{
    m_in_destruction = true;
    m_available_indices = std::queue<size_t>();
    m_all_frames.clear();
}

void MediaFramePool::return_frame_to_pool(size_t index)
{
    if (m_in_destruction || m_stop) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    m_available_indices.push(index);
    m_cv.notify_one();
}

std::shared_ptr<MediaFrame> MediaFramePool::get_frame()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_available_indices.empty() || m_stop) {
        return nullptr;
    }

    size_t index = m_available_indices.front();
    m_available_indices.pop();

    return std::shared_ptr<MediaFrame>(
        &m_all_frames[index],
        [this, index](MediaFrame*) { this->return_frame_to_pool(index); }
    );
}

std::shared_ptr<MediaFrame> MediaFramePool::get_frame_blocking()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [this] { return !m_available_indices.empty() || m_stop; });

    if (m_stop) {
        return nullptr;
    }

    size_t index = m_available_indices.front();
    m_available_indices.pop();

    return std::shared_ptr<MediaFrame>(
        &m_all_frames[index],
        [this, index](MediaFrame*) { this->return_frame_to_pool(index); }
    );
}

size_t MediaFramePool::get_available_frames_count() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_available_indices.size();
}

void MediaFramePool::stop()
{
    m_stop = true;
    m_cv.notify_all();
}
