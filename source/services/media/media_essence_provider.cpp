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

#include <cstring>
#include <cstdlib>
#include <random>

#include "rdk/services/media/media_essence_provider.h"
#include "rdk/services/media/media_settings_video.h"

using namespace rivermax::dev_kit::services;

MediaUnitBuffer::MediaUnitBuffer(size_t buffer_size) :
    m_owned_buffer(new byte_t[buffer_size]),
    m_buffer_ptr(m_owned_buffer.get()),
    m_size(buffer_size),
    m_is_owned(true),
    m_memory_location(MemoryLocation::Host)
{
}

MediaUnitBuffer::MediaUnitBuffer(byte_t* external_buffer, size_t buffer_size,
    MemoryLocation memory_location) :
    m_owned_buffer(nullptr),
    m_buffer_ptr(external_buffer),
    m_size(buffer_size),
    m_is_owned(false),
    m_memory_location(memory_location)
{
}

MediaUnitBuffer::MediaUnitBuffer(const std::shared_ptr<byte_t>& shared_buffer, size_t buffer_size,
    MemoryLocation memory_location) :
    m_owned_buffer(nullptr),
    m_buffer_ptr(shared_buffer.get()),
    m_size(buffer_size),
    m_is_owned(false),
    m_memory_location(memory_location)
{
}

MediaUnitBuffer::MediaUnitBuffer(MediaUnitBuffer&& other) noexcept :
    m_owned_buffer(std::move(other.m_owned_buffer)),
    m_buffer_ptr(other.m_buffer_ptr),
    m_size(other.m_size),
    m_is_owned(other.m_is_owned),
    m_memory_location(other.m_memory_location)
{
    // If other owned its buffer, its pointer is now invalid after the move.
    if (other.m_is_owned) {
        other.m_buffer_ptr = nullptr;
    }
}

MediaUnitBuffer& MediaUnitBuffer::operator=(MediaUnitBuffer&& other) noexcept
{
    if (this != &other) {
        m_owned_buffer = std::move(other.m_owned_buffer);
        m_buffer_ptr = other.m_buffer_ptr;
        m_size = other.m_size;
        m_is_owned = other.m_is_owned;
        m_memory_location = other.m_memory_location;

        // If other owned its buffer, its pointer is now invalid after the move
        if (other.m_is_owned) {
            other.m_buffer_ptr = nullptr;
        }
    }
    return *this;
}

MediaUnit::MediaUnit(size_t buffer_size) :
    data(std::make_unique<MediaUnitBuffer>(buffer_size)),
    metadata(nullptr)
{
}

MediaUnit::MediaUnit(byte_t* external_buffer, size_t buffer_size,
    MemoryLocation memory_location) :
    data(std::make_unique<MediaUnitBuffer>(external_buffer, buffer_size, memory_location)),
    metadata(nullptr)
{
}

MediaUnit::MediaUnit(const std::shared_ptr<byte_t>& shared_buffer, size_t buffer_size,
    MemoryLocation memory_location) :
    data(std::make_unique<MediaUnitBuffer>(shared_buffer, buffer_size, memory_location)),
    metadata(nullptr)
{
}

MediaUnit::MediaUnit(std::unique_ptr<IMediaUnitBuffer>&& unit_buffer) :
    data(std::move(unit_buffer)),
    metadata(nullptr)
{
}

NullEssenceProvider::NullEssenceProvider(const MediaSettings& media_settings) :
    m_media_settings(media_settings),
    m_media_unit_not_available_probability(0.0f)
{
    set_media_unit_settings(m_cached_metadata, m_cached_media_unit_size);
}

void NullEssenceProvider::set_media_unit_settings(MediaUnitMetadata& metadata, size_t& media_unit_size)
{
    auto& video_settings = static_cast<const SMPTE_2110_20_MediaSettings&>(m_media_settings);
    media_unit_size = video_settings.bytes_per_frame;
    metadata.resolution = video_settings.resolution;
    metadata.smpte_standard = video_settings.get_smpte_standard();
}

std::shared_ptr<MediaUnit> NullEssenceProvider::get_media_unit_blocking()
{
    auto media_unit = std::make_shared<MediaUnit>(m_cached_media_unit_size);
    media_unit->add_metadata(m_cached_metadata);
    return media_unit;
}

std::shared_ptr<MediaUnit> NullEssenceProvider::get_media_unit_non_blocking()
{
    return is_media_unit_available() ? get_media_unit_blocking() : nullptr;
}

ReturnStatus NullEssenceProvider::set_media_unit_not_available_probability(float probability)
{
    if (probability < 0.0f || probability > 1.0f) {
        std::cerr << "Invalid probability value [0..1.0]: " << probability << std::endl;
        return ReturnStatus::failure;
    }
    m_media_unit_not_available_probability = probability;
    return ReturnStatus::success;
}

bool NullEssenceProvider::is_media_unit_available() const
{
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    return distribution(generator) >= m_media_unit_not_available_probability;
}

BufferedEssenceProvider::BufferedEssenceProvider(size_t max_queue_size) :
    m_max_queue_size(max_queue_size),
    m_stop(false)
{
}

BufferedEssenceProvider::~BufferedEssenceProvider()
{
    stop();
}

std::shared_ptr<MediaUnit> BufferedEssenceProvider::get_media_unit_blocking()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    // Wait until a media unit is available or stop is requested
    m_cv.wait(lock, [this] { return !m_media_unit_queue.empty() || m_stop; });
    if (m_stop && m_media_unit_queue.empty()) {
        return nullptr;
    }
    auto media_unit = m_media_unit_queue.front();
    m_media_unit_queue.pop();
    return media_unit;
}

std::shared_ptr<MediaUnit> BufferedEssenceProvider::get_media_unit_non_blocking()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_media_unit_queue.empty()) {
        return nullptr;
    }
    auto media_unit = m_media_unit_queue.front();
    m_media_unit_queue.pop();
    return media_unit;
}

ReturnStatus BufferedEssenceProvider::add_media_unit(std::shared_ptr<MediaUnit> media_unit)
{
    if (!media_unit) {
        std::cerr << "Received null media unit" << std::endl;
        return ReturnStatus::failure;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_max_queue_size > 0 && m_media_unit_queue.size() >= m_max_queue_size) {
        return ReturnStatus::frame_send_queue_full;
    }
    m_media_unit_queue.push(std::move(media_unit));
    m_cv.notify_one();
    return ReturnStatus::success;
}

size_t BufferedEssenceProvider::get_queue_size() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_media_unit_queue.size();
}

void BufferedEssenceProvider::stop()
{
    m_stop = true;
    m_cv.notify_all();
}

MediaFileEssenceProvider::MediaFileEssenceProvider(const std::string& file_path, SMPTEStandard smpte_standard,
                                                   size_t media_unit_size,
                                                   MemoryAllocator& mem_allocator, bool loop)
    : m_file_path(file_path)
    , m_mem_allocator(mem_allocator)
    , m_smpte_standard(smpte_standard)
    , m_media_unit_size(media_unit_size)
    , m_aligned_media_unit_size(mem_allocator.align_length(media_unit_size))
    , m_loop_media_units(loop)
    , m_stop(false)
    , m_media_units_loaded(false)
{
}

MediaFileEssenceProvider::~MediaFileEssenceProvider()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
    }
    m_cv.notify_all();
    if (m_input_file.is_open()) {
        m_input_file.close();
    }
}

std::shared_ptr<MediaUnit> MediaFileEssenceProvider::get_media_unit_blocking()
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

std::shared_ptr<MediaUnit> MediaFileEssenceProvider::get_media_unit_non_blocking()
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

ReturnStatus MediaFileEssenceProvider::allocate_media_units_memory(size_t file_size,
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

ReturnStatus MediaFileEssenceProvider::read_media_units(byte_t* file_memory_buffer)
{
    size_t unit_index = 0;
    byte_t* cur_unit_ptr = file_memory_buffer;
    auto temp_buffer = std::make_unique<byte_t[]>(m_media_unit_size);
    auto mem_utils = m_mem_allocator.get_memory_utils();
    if (!mem_utils) {
        std::cerr << "Failed to get memory utils" << std::endl;
        return ReturnStatus::failure;
    }

    while (true) {
        m_input_file.read(reinterpret_cast<char*>(temp_buffer.get()), m_media_unit_size);
        std::streamsize bytes_read = m_input_file.gcount();

        if (bytes_read == 0) {
            // If we reached EOF, break out normally.
            if (m_input_file.eof()) {
                break;
            }
            // If not EOF but still no data read, then an error occurred.
            if (m_input_file.fail() && !m_input_file.eof()) {
                std::cerr << "Failed to read media unit: " << unit_index << " from file: " << m_file_path << std::endl;
                return ReturnStatus::failure;
            }
        }

        if (bytes_read < static_cast<std::streamsize>(m_media_unit_size)) {
            break;
        }

        mem_utils->memory_copy(cur_unit_ptr, temp_buffer.get(), m_media_unit_size);
        auto media_unit = std::make_shared<MediaUnit>(cur_unit_ptr, m_media_unit_size);
        m_media_unit_queue.push(media_unit);

        m_cv.notify_one();
        unit_index++;
        cur_unit_ptr += m_aligned_media_unit_size;
    }
    return ReturnStatus::success;
}

ReturnStatus MediaFileEssenceProvider::load_media_units()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_media_units_loaded) {
        return ReturnStatus::success;
    }
    m_input_file.open(m_file_path, std::ios::binary);
    if (!m_input_file.is_open()) {
        std::cerr << "Failed to open file: " << m_file_path << std::endl;
        return ReturnStatus::failure;
    }

    m_input_file.clear();
    m_input_file.seekg(0, std::ios::end);
    size_t file_size = m_input_file.tellg();
    m_input_file.seekg(0, std::ios::beg);

    byte_t* file_memory_buffer = nullptr;
    size_t required_memory_size = 0;
    auto rc = allocate_media_units_memory(file_size, file_memory_buffer, required_memory_size);
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

void MediaFileEssenceProvider::handle_looping_media_unit(std::shared_ptr<MediaUnit>& media_unit)
{
    if (m_loop_media_units && media_unit) {
        m_media_unit_queue.push(media_unit);
    }
    m_cv.notify_one();
}

void MediaFileEssenceProvider::stop()
{
    m_stop = true;
    m_cv.notify_all();
}
