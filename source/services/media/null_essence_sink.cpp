/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <random>

#include "rdk/services/media/null_essence_sink.h"

using namespace rivermax::dev_kit::services;

ReturnStatus NullEssenceSink::put_media_unit_blocking(std::shared_ptr<MediaUnit> media_unit)
{
    if (media_unit == nullptr) {
        std::cerr << "Failed to put media unit (blocking): media_unit is nullptr." << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

ReturnStatus NullEssenceSink::put_media_unit_non_blocking(std::shared_ptr<MediaUnit> media_unit)
{
    if (media_unit == nullptr) {
        std::cerr << "Failed to put media unit (non-blocking): media_unit is nullptr." << std::endl;
        return ReturnStatus::failure;
    }
    return can_accept_media_unit() ? ReturnStatus::success : ReturnStatus::sink_not_ready;
}

ReturnStatus NullEssenceSink::set_put_failure_probability(float probability)
{
    if (probability < 0.0f || probability > 1.0f) {
        std::cerr << "Invalid probability value [0..1.0]: " << probability << std::endl;
        return ReturnStatus::failure;
    }
    m_put_failure_probability = probability;
    return ReturnStatus::success;
}

bool NullEssenceSink::can_accept_media_unit() const
{
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    return distribution(generator) >= m_put_failure_probability;
}
