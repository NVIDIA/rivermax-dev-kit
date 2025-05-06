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

#include "rdk/services/media/media_frame_provider.h"

using namespace rivermax::dev_kit::services;

FrameBuffer::FrameBuffer(size_t buffer_size) :
    owned_buffer(new byte_t[buffer_size]),
    buffer_ptr(owned_buffer.get()),
    size(buffer_size),
    is_owned(true)
{
}

FrameBuffer::FrameBuffer(byte_t* external_buffer, size_t buffer_size) :
    owned_buffer(nullptr),
    buffer_ptr(external_buffer),
    size(buffer_size),
    is_owned(false)
{
}

FrameBuffer::FrameBuffer(const std::shared_ptr<byte_t>& shared_buffer, size_t buffer_size) :
    owned_buffer(nullptr),
    buffer_ptr(shared_buffer.get()),
    size(buffer_size),
    is_owned(false)
{
}

FrameBuffer::FrameBuffer(FrameBuffer&& other) noexcept :
    owned_buffer(std::move(other.owned_buffer)),
    buffer_ptr(other.buffer_ptr),
    size(other.size),
    is_owned(other.is_owned)
{
    // If other owned its buffer, its pointer is now invalid after the move.
    if (other.is_owned) {
        other.buffer_ptr = nullptr;
    }
}

FrameBuffer& FrameBuffer::operator=(FrameBuffer&& other) noexcept
{
    if (this != &other) {
        owned_buffer = std::move(other.owned_buffer);
        buffer_ptr = other.buffer_ptr;
        size = other.size;
        is_owned = other.is_owned;

        // If other owned its buffer, its pointer is now invalid after the move
        if (other.is_owned) {
            other.buffer_ptr = nullptr;
        }
    }
    return *this;
}

MediaFrame::MediaFrame(size_t buffer_size) :
    data(buffer_size),
    metadata(nullptr)
{
}

MediaFrame::MediaFrame(byte_t* external_buffer, size_t buffer_size) :
    data(external_buffer, buffer_size),
    metadata(nullptr)
{
}

MediaFrame::MediaFrame(const std::shared_ptr<byte_t>& shared_buffer, size_t buffer_size) :
    data(shared_buffer, buffer_size),
    metadata(nullptr)
{
}

NullFrameProvider::NullFrameProvider(const MediaSettings& media_settings) :
    m_media_settings(media_settings),
    m_frame_not_available_probability(0.0f)
{
    set_frame_settings(m_cached_metadata, m_cached_frame_size);
}

void NullFrameProvider::set_frame_settings(FrameMetadata& metadata, size_t& frame_size)
{
    frame_size = m_media_settings.bytes_per_frame;
    metadata.resolution = m_media_settings.resolution;
    metadata.media_type = MediaType::Video;
}

std::shared_ptr<MediaFrame> NullFrameProvider::get_frame_blocking()
{
    auto frame = std::make_shared<MediaFrame>(m_cached_frame_size);
    frame->add_metadata(m_cached_metadata);
    return frame;
}

std::shared_ptr<MediaFrame> NullFrameProvider::get_frame_not_blocking()
{
    return is_frame_available() ? get_frame_blocking() : nullptr;
}

ReturnStatus NullFrameProvider::set_frame_not_available_probability(float probability)
{
    if (probability < 0.0f || probability > 1.0f) {
        std::cerr << "Invalid probability value [0..1.0]: " << probability << std::endl;
        return ReturnStatus::failure;
    }
    m_frame_not_available_probability = probability;
    return ReturnStatus::success;
}

bool NullFrameProvider::is_frame_available() const
{
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    return distribution(generator) >= m_frame_not_available_probability;
}

BufferedMediaFrameProvider::BufferedMediaFrameProvider(size_t max_queue_size) :
    m_max_queue_size(max_queue_size),
    m_stop(false)
{
}

BufferedMediaFrameProvider::~BufferedMediaFrameProvider()
{
    stop();
}

std::shared_ptr<MediaFrame> BufferedMediaFrameProvider::get_frame_blocking()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    // Wait until a frame is available or stop is requested
    m_cv.wait(lock, [this] { return !m_frame_queue.empty() || m_stop; });
    if (m_stop && m_frame_queue.empty()) {
        return nullptr;
    }
    auto frame = m_frame_queue.front();
    m_frame_queue.pop();
    return frame;
}

std::shared_ptr<MediaFrame> BufferedMediaFrameProvider::get_frame_not_blocking()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_frame_queue.empty()) {
        return nullptr;
    }
    auto frame = m_frame_queue.front();
    m_frame_queue.pop();
    return frame;
}

ReturnStatus BufferedMediaFrameProvider::add_frame(std::shared_ptr<MediaFrame> frame)
{
    if (!frame) {
        return ReturnStatus::failure;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_max_queue_size > 0 && m_frame_queue.size() >= m_max_queue_size) {
        return ReturnStatus::frame_send_queue_full;
    }
    m_frame_queue.push(std::move(frame));
    m_cv.notify_one();
    return ReturnStatus::success;
}

size_t BufferedMediaFrameProvider::get_queue_size() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_frame_queue.size();
}

void BufferedMediaFrameProvider::stop()
{
    m_stop = true;
    m_cv.notify_all();
}

MediaFileFrameProvider::MediaFileFrameProvider(const std::string &file_path, MediaType type,
    size_t frame_size, MemoryAllocator& mem_allocator, bool loop) :
    m_file_path(file_path),
    m_mem_allocator(mem_allocator),
    m_media_type(type),
    m_frame_size(frame_size),
    m_loop_frames(loop),
    m_stop(false),
    m_frames_loaded(false)
{
}

MediaFileFrameProvider::~MediaFileFrameProvider()
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

std::shared_ptr<MediaFrame> MediaFileFrameProvider::get_frame_blocking()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [this] { return !m_frame_queue.empty() || m_frames_loaded || m_stop; });
    if (m_stop && m_frame_queue.empty()) {
        return nullptr;
    }
    auto frame = m_frame_queue.front();
    m_frame_queue.pop();
    handle_looping_frame(frame);
    return frame;
}

std::shared_ptr<MediaFrame> MediaFileFrameProvider::get_frame_not_blocking()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_frame_queue.empty()) {
        return nullptr;
    }
    auto frame = m_frame_queue.front();
    m_frame_queue.pop();
    handle_looping_frame(frame);
    return frame;
}

ReturnStatus MediaFileFrameProvider::allocate_frames_memory(size_t file_size,
    byte_t*& file_memory_buffer, size_t& required_memory_size)
{
    size_t num_frames = file_size / m_frame_size;
    required_memory_size = num_frames * m_frame_size;
    required_memory_size = m_mem_allocator.align_length(required_memory_size);

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

ReturnStatus MediaFileFrameProvider::read_frames(byte_t* file_memory_buffer)
{
    size_t frame_index = 0;
    byte_t* cur_frame_ptr = file_memory_buffer;
    auto temp_buffer = std::make_unique<byte_t[]>(m_frame_size);
    auto mem_utils = m_mem_allocator.get_memory_utils();
    if (!mem_utils) {
        std::cerr << "Failed to get memory utils" << std::endl;
        return ReturnStatus::failure;
    }

    while (true) {
        m_input_file.read(reinterpret_cast<char*>(temp_buffer.get()), m_frame_size);
        std::streamsize bytes_read = m_input_file.gcount();

        if (bytes_read == 0) {
            // If we reached EOF, break out normally.
            if (m_input_file.eof()) {
                break;
            }
            // If not EOF but still no data read, then an error occurred.
            if (m_input_file.fail() && !m_input_file.eof()) {
                std::cerr << "Failed to read frame: " << frame_index << " from file: " << m_file_path << std::endl;
                return ReturnStatus::failure;
            }
        }

        if (bytes_read < static_cast<std::streamsize>(m_frame_size)) {
            break;
        }

        mem_utils->memory_copy(cur_frame_ptr, temp_buffer.get(), m_frame_size);
        auto frame = std::make_shared<MediaFrame>(cur_frame_ptr, m_frame_size);
        m_frame_queue.push(frame);

        m_cv.notify_one();
        frame_index++;
        cur_frame_ptr += m_frame_size;
    }
    return ReturnStatus::success;
}

ReturnStatus MediaFileFrameProvider::load_frames()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_frames_loaded) {
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
    auto rc = allocate_frames_memory(file_size, file_memory_buffer, required_memory_size);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to allocate memory for file: " << m_file_path << std::endl;
        return rc;
    }

    rc = read_frames(file_memory_buffer);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to read frames from file: " << m_file_path << std::endl;
        return rc;
    }

    m_frames_loaded = true;
    m_cv.notify_all();
    return ReturnStatus::success;
}

void MediaFileFrameProvider::handle_looping_frame(std::shared_ptr<MediaFrame>& frame)
{
    if (m_loop_frames && frame) {
        m_frame_queue.push(frame);
    }
    m_cv.notify_one();
}

void MediaFileFrameProvider::stop()
{
    m_stop = true;
    m_cv.notify_all();
}
