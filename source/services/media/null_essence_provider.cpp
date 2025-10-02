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

#include "rdk/services/media/null_essence_provider.h"
#include "rdk/services/media/media_settings_video.h"

using namespace rivermax::dev_kit::services;


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
