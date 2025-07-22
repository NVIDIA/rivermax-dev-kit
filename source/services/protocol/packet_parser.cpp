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

#include "rdk/services/protocol/packet_parser.h"

using namespace rivermax::dev_kit::services;

void PacketParser::recompute_offsets()
{
    m_offsets.clear();
    m_offsets.reserve(m_parsers.size());
    m_header_accessors.clear();

    size_t current_offset = 0;
    for (size_t i = 0; i < m_parsers.size(); ++i) {
        m_offsets.push_back(current_offset);
        size_t header_size = m_parsers[i]->get_header_size(nullptr);
        current_offset += header_size;
    }
    m_min_packet_size = current_offset;
}

bool PacketParser::validate(const byte_t* data, size_t length) const
{
    if (length < m_min_packet_size) {
        return false;
    }

    size_t current_offset = 0;
    for (size_t i = 0; i < m_parsers.size(); ++i) {
        size_t remaining_bytes = length - current_offset;

        if (!m_parsers[i]->validate(data + current_offset, remaining_bytes)) {
            return false;
        }

        size_t actual_header_size = m_parsers[i]->get_header_size(data + current_offset);
        current_offset += actual_header_size;

        if (current_offset > length) {
            return false;
        }
    }
    return true;
}
