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

#include <cstdio>

#include "rdk/services/media/closed_caption_mock_source.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

std::string ClosedCaptionMockSource::get_caption_text(uint64_t timestamp_ns)
{
    // Calculate position within 4-second cycle
    uint64_t cycle_time = timestamp_ns % CYCLE_DURATION_NS;
    bool is_caption_on = (cycle_time < ON_DURATION_NS);

    if (is_caption_on) {
        // Calculate cycle number directly from timestamp
        uint32_t cycle_number = static_cast<uint32_t>(timestamp_ns / CYCLE_DURATION_NS);

        // Generate caption text with cycle number
        char text_buffer[32];
        snprintf(text_buffer, sizeof(text_buffer), "CAPTION %02u", cycle_number % 100);
        return std::string(text_buffer);
    }

    return "";
}

} // namespace services
} // namespace dev_kit
} // namespace rivermax
