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

#include <cstdlib>
#include <cstring>
#include <random>

#include "rdk/services/media/media_file_essence_source.h"
#include "rdk/services/media/media_settings_video.h"

using namespace rivermax::dev_kit::services;

MediaFileEssenceSource::MediaFileEssenceSource(const std::string &file_path, SMPTEStandard smpte_standard,
    size_t media_unit_size, MemoryAllocator& mem_allocator, bool loop) :
    m_file_path(file_path),
    m_mem_allocator(mem_allocator),
    m_smpte_standard(smpte_standard),
    m_media_unit_size(media_unit_size),
    m_aligned_media_unit_size(mem_allocator.align_length(media_unit_size)),
    m_loop_media_units(loop),
    m_stop(false),
    m_media_units_loaded(false),
    m_file_reader(file_path, mem_allocator.get_memory_utils(), false)
{
}

MediaFileEssenceSource::~MediaFileEssenceSource()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
    }
    m_cv.notify_all();
}

ReturnStatus MediaFileEssenceSource::get_data_size(size_t& data_size) const
{
    return m_file_reader.get_file_size(data_size);
}

std::shared_ptr<MediaUnit> MediaFileEssenceSource::get_media_unit_blocking()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [this] { return !m_media_unit_queue.empty() || m_media_units_loaded || m_stop; });
    if (m_stop && m_media_unit_queue.empty()) {
        return nullptr;
    }
    auto media_unit = m_media_unit_queue.front();
    m_media_unit_queue.pop();
    handle_looping_media_unit(media_unit);
    return media_unit;
}

std::shared_ptr<MediaUnit> MediaFileEssenceSource::get_media_unit_non_blocking()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_media_unit_queue.empty()) {
        return nullptr;
    }
    auto media_unit = m_media_unit_queue.front();
    m_media_unit_queue.pop();
    handle_looping_media_unit(media_unit);
    return media_unit;
}

ReturnStatus MediaFileEssenceSource::allocate_media_units_memory(size_t file_size,
    byte_t*& file_memory_buffer, size_t& required_memory_size)
{
    size_t num_units = file_size / m_media_unit_size;
    required_memory_size = num_units * m_aligned_media_unit_size;
    file_memory_buffer = static_cast<byte_t*>(m_mem_allocator.allocate_aligned(required_memory_size,
        m_mem_allocator.get_page_size()));
    if (!file_memory_buffer) {
        std::cerr << "Failed to allocate memory for file: " << m_file_path
            << " required size: " << required_memory_size << std::endl;
        return ReturnStatus::failure;
    } else {
        std::cout << "Allocated memory for file: " << m_file_path
            << " required size: " << required_memory_size << std::endl;
    }

    return ReturnStatus::success;
}

ReturnStatus MediaFileEssenceSource::read_media_units(byte_t* file_memory_buffer)
{
    size_t unit_index = 0;
    byte_t* cur_unit_ptr = file_memory_buffer;
    auto mem_utils = m_mem_allocator.get_memory_utils();
    if (!mem_utils) {
        std::cerr << "Failed to get memory utils" << std::endl;
        return ReturnStatus::failure;
    }

    while (true) {
        size_t bytes_read = 0;
        auto rc = m_file_reader.read_and_copy(cur_unit_ptr, m_media_unit_size, bytes_read);
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to read media unit: " << unit_index << " from file: " << m_file_path << std::endl;
            return rc;
        }

        if (bytes_read == 0 || bytes_read < m_media_unit_size) {
            break;  // EOF or partial read
        }

        auto media_unit = std::make_shared<MediaUnit>(cur_unit_ptr, m_media_unit_size, m_smpte_standard);
        m_media_unit_queue.push(std::move(media_unit));

        m_cv.notify_one();
        unit_index++;
        cur_unit_ptr += m_aligned_media_unit_size;
    }
    return ReturnStatus::success;
}

ReturnStatus MediaFileEssenceSource::load_media_units()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_media_units_loaded) {
        return ReturnStatus::success;
    }

    auto rc = m_file_reader.open();
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to open file: " << m_file_path << std::endl;
        return rc;
    }

    size_t file_size = 0;
    rc = m_file_reader.get_file_size(file_size);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to get file size: " << m_file_path << std::endl;
        return rc;
    }

    byte_t* file_memory_buffer = nullptr;
    size_t required_memory_size = 0;
    rc = allocate_media_units_memory(file_size, file_memory_buffer, required_memory_size);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to allocate memory for file: " << m_file_path << std::endl;
        return rc;
    }

    rc = read_media_units(file_memory_buffer);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to read units from file: " << m_file_path << std::endl;
        return rc;
    }

    m_media_units_loaded = true;
    m_cv.notify_all();
    return ReturnStatus::success;
}

void MediaFileEssenceSource::handle_looping_media_unit(std::shared_ptr<MediaUnit>& media_unit)
{
    if (m_loop_media_units && media_unit) {
        m_media_unit_queue.push(media_unit);
    }
    m_cv.notify_one();
}

void MediaFileEssenceSource::stop()
{
    m_stop = true;
    m_cv.notify_all();
}
