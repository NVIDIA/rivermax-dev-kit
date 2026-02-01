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

#include "rdk/services/media/buffered_essence_source.h"
#include "rdk/services/media/media_settings_video.h"

using namespace rivermax::dev_kit::services;

BufferedEssenceSource::BufferedEssenceSource(size_t max_queue_size) :
    m_max_queue_size(max_queue_size),
    m_stop(false)
{
}

BufferedEssenceSource::~BufferedEssenceSource()
{
    stop();
}

std::shared_ptr<MediaUnit> BufferedEssenceSource::get_media_unit_blocking()
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

std::shared_ptr<MediaUnit> BufferedEssenceSource::get_media_unit_non_blocking()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_media_unit_queue.empty()) {
        return nullptr;
    }
    auto media_unit = m_media_unit_queue.front();
    m_media_unit_queue.pop();
    return media_unit;
}

ReturnStatus BufferedEssenceSource::add_media_unit(std::shared_ptr<MediaUnit> media_unit)
{
    if (!media_unit) {
        std::cerr << "Received null media unit" << std::endl;
        return ReturnStatus::failure;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_max_queue_size > 0 && m_media_unit_queue.size() >= m_max_queue_size) {
        return ReturnStatus::media_unit_send_queue_full;
    }
    m_media_unit_queue.push(std::move(media_unit));
    m_cv.notify_one();
    return ReturnStatus::success;
}

size_t BufferedEssenceSource::get_queue_size() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_media_unit_queue.size();
}

void BufferedEssenceSource::stop()
{
    m_stop = true;
    m_cv.notify_all();
}
