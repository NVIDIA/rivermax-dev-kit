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

#include <chrono>
#include <iostream>
#include <thread>

#include "rdk/services/media/media_file_streaming_provider.h"
#include "rdk/services/error_handling/error_handling.h"
#include "rdk/services/media/buffered_essence_source.h"
#include "rdk/services/media/media_unit_pool.h"

using namespace rivermax::dev_kit::services;

MediaFileStreamingProvider::MediaFileStreamingProvider(
    const std::string& file_path, SMPTEStandard smpte_standard, size_t media_unit_size,
    std::shared_ptr<BufferedEssenceSource> essence_source,
    std::shared_ptr<MemoryAllocator> memory_allocator, bool loop,
    size_t sleep_duration_microseconds) :
    m_file_path(file_path),
    m_smpte_standard(smpte_standard),
    m_media_unit_size(media_unit_size),
    m_essence_source(std::move(essence_source)),
    m_memory_utils(memory_allocator->get_memory_utils()),
    m_memory_allocator(std::move(memory_allocator)),
    m_loop_media_units(loop),
    m_stop(false),
    m_sleep_duration_microseconds(sleep_duration_microseconds),
    m_file_reader(file_path, m_memory_utils, loop)
{
    if (!m_memory_allocator) {
        std::cerr << "MemoryAllocator is not valid" << std::endl;
        throw std::runtime_error("MemoryAllocator is not valid");
    }

    m_media_unit_pool = std::make_unique<MediaUnitPool>(
        MEMORY_POOL_MEDIA_UNIT_COUNT, m_media_unit_size, m_smpte_standard, *m_memory_allocator);
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

    auto rc = m_file_reader.open();
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to open file: " << m_file_path << std::endl;
        return rc;
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

    while (!m_stop && SignalHandler::get_received_signal() < 0) {
        // Get a media unit from the pool
        auto media_unit = m_media_unit_pool->get_media_unit();
        if (!media_unit) {
            std::this_thread::sleep_for(std::chrono::microseconds(m_sleep_duration_microseconds));
            continue;
        }

        // Read data directly into unit buffer using @ref MediaFileReader
        size_t bytes_read = 0;
        auto rc = m_file_reader.read_and_copy(media_unit->data->get(), m_media_unit_size, bytes_read);

        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to read media unit from file: " << m_file_path << std::endl;
            break;
        }

        if (bytes_read == 0) {
            if (m_file_reader.is_eof()) {
                if (m_file_reader.handle_eof()) {
                    continue;  // Looping, continue reading
                }
                break;  // No loop, we're done
            }
        }

        while (!m_stop && SignalHandler::get_received_signal() < 0) {
            if (m_essence_source->add_media_unit(media_unit) == ReturnStatus::success) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(m_sleep_duration_microseconds));
        }

        if (SignalHandler::get_received_signal() >= 0) {
            break;
        }
    }

    m_essence_source->stop();
    m_media_unit_pool->stop();
    m_initialized = false;
}
