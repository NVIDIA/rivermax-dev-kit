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

#include <iostream>
#include <chrono>
#include <thread>

#include "rdk/services/error_handling/error_handling.h"
#include "rdk/services/media/media_file_streaming_provider.h"

using namespace rivermax::dev_kit::services;

MediaFileStreamingProvider::MediaFileStreamingProvider(const std::string& file_path,
    MediaType media_type, size_t frame_size, std::shared_ptr<BufferedMediaFrameProvider> frame_provider,
    std::shared_ptr<MemoryAllocator> memory_allocator, bool loop,
    size_t sleep_duration_microseconds) :
    m_file_path(file_path),
    m_media_type(media_type),
    m_frame_size(frame_size),
    m_frame_provider(std::move(frame_provider)),
    m_memory_allocator(std::move(memory_allocator)),
    m_loop_frames(loop),
    m_stop(false),
    m_sleep_duration_microseconds(sleep_duration_microseconds)
{
    if (!m_memory_allocator) {
        std::cerr << "MemoryAllocator is not valid" << std::endl;
        throw std::runtime_error("MemoryAllocator is not valid");
    }

    m_memory_utils = m_memory_allocator->get_memory_utils();
    m_frame_pool = std::make_unique<MediaFramePool>(
        MEMORY_POOL_FRAME_COUNT, m_frame_size, *m_memory_allocator);
}

MediaFileStreamingProvider::~MediaFileStreamingProvider()
{
    stop();
}

ReturnStatus MediaFileStreamingProvider::initialize()
{
    if (m_initialized) {
        return ReturnStatus::success;
    }

    m_input_file.open(m_file_path, std::ios::binary);
    if (!m_input_file.is_open()) {
        std::cerr << "Failed to open file: " << m_file_path << std::endl;
        return ReturnStatus::failure;
    }

    if (!m_memory_utils) {
        std::cerr << "Failed to get memory utils" << std::endl;
        return ReturnStatus::failure;
    }

    m_initialized = true;
    return ReturnStatus::success;
}

void MediaFileStreamingProvider::stop()
{
    m_stop = true;
    m_cv.notify_all();
}

void MediaFileStreamingProvider::operator()()
{
    if (!m_initialized) {
        std::cerr << "MediaFileStreamingProvider is not initialized" << std::endl;
        return;
    }
    if (!m_input_file.is_open()) {
        std::cerr << "File is not open: " << m_file_path << std::endl;
        return;
    }

    // Create a temporary buffer for reading from file
    auto temp_buffer = std::make_unique<byte_t[]>(m_frame_size);

    while (!m_stop && SignalHandler::get_received_signal() < 0) {
        // Get a frame from the pool
        auto frame = m_frame_pool->get_frame();
        if (!frame) {
            std::this_thread::sleep_for(std::chrono::microseconds(m_sleep_duration_microseconds));
            continue;
        }

        // Read data into temporary buffer first
        m_input_file.read(reinterpret_cast<char*>(temp_buffer.get()), m_frame_size);
        std::streamsize bytes_read = m_input_file.gcount();

        if (bytes_read == 0) {
            if (m_input_file.eof()) {
                if (!m_loop_frames) {
                    break;
                }
                // Loop the file, start reading from the beginning
                m_input_file.clear();
                m_input_file.seekg(0, std::ios::beg);
                continue;
            } else if (m_input_file.fail()) {
                std::cerr << "Failed to read frame from file: " << m_file_path << std::endl;
                break;
            }
        }

        // Copy from temporary buffer to frame buffer using memory utils
        m_memory_utils->memory_copy(frame->data.get(), temp_buffer.get(),
            bytes_read < static_cast<std::streamsize>(m_frame_size) ? bytes_read : m_frame_size);

        // Handle partial frame read if needed
        if (bytes_read < static_cast<std::streamsize>(m_frame_size)) {
            m_memory_utils->memory_set(frame->data.get() + bytes_read, 0, m_frame_size - bytes_read);
        }

        while (!m_stop && SignalHandler::get_received_signal() < 0) {
            if (m_frame_provider->add_frame(frame) == ReturnStatus::success) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(m_sleep_duration_microseconds));
        }

        if (SignalHandler::get_received_signal() >= 0) {
            break;
        }
    }

    m_input_file.close();
    m_frame_provider->stop();
    m_frame_pool->stop();
    m_initialized = false;
}
