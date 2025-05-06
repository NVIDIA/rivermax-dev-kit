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

#include <cstddef>

#include "rdk/core/chunk/chunk_interface.h"
#include "rdk/core/chunk/generic_chunk.h"
#include "rdk/io_node/common/generic_buffer_writer.h"

using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

void GenericBufferWriter::write_buffer(IChunk& chunk)
{
    auto& app_chunk = dynamic_cast<GenericChunk&>(chunk);

    for (size_t packet_index = 0; packet_index < app_chunk.get_length(); packet_index++) {
        auto& packet = app_chunk.get_packet(packet_index);
        for (auto& iov_elem : packet) {
            m_mem_utils->memory_set(reinterpret_cast<void*>(iov_elem.addr), '0' + (packet_index % 9), iov_elem.length);
        }
    }
}
