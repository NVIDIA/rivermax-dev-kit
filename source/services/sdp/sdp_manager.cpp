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

#include <chrono>
#include <string>
#include <sstream>

#include "sdptransform.hpp"

#include "rdk/services/sdp/sdp_manager.h"

using namespace rivermax::dev_kit::services;

SDPManager::operator json() const
{
    json sdp;

    sdp.merge_patch(*m_session_description);
    sdp.merge_patch(*m_time_description);

    for (const auto& media_description : m_media_descriptions) {
        sdp["media"].push_back(*media_description);
    }

    return sdp;
}

SDPManager::operator std::string() const
{
    json sdp = this->operator json();
    return sdptransform::write(sdp);
}

size_t SDPManager::generate_ntp_id(size_t offset_seconds)
{
    auto now_ntp =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) + 2208988800U + offset_seconds;
    std::ostringstream oss;
    oss << std::setw(10) << std::setfill('0') << now_ntp;

    return std::stoul(oss.str());
}
