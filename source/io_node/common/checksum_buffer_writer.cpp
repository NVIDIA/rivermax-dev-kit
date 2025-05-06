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

#include "rdk/io_node/common/checksum_buffer_writer.h"
#include "rdk/core/chunk/chunk_interface.h"
#include "rdk/core/chunk/generic_chunk.h"

using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

void ChecksumBufferWriter::write_buffer(IChunk& chunk)
{
    auto& app_chunk = static_cast<GenericChunk&>(chunk);

    for (size_t packet_index = 0; packet_index < app_chunk.get_length(); packet_index++) {
        auto& packet = app_chunk.get_packet(packet_index);
        for (auto& iov_elem : packet) {
            /*
             * Fill the data with a randomly repeating char.
             *
             * Coverity warns about rand() usage for security or cryptography related usage
             * This case is just to pick an arbitrary character and fill data with it so
             * so we voluntarily silence coverity warning.
             */
            // coverity[dont_call]
            unsigned char data_char = rand() % 255;
            m_mem_utils->memory_set(reinterpret_cast<void*>(
                reinterpret_cast<uint8_t*>(iov_elem.addr) + iov_elem.length), data_char, iov_elem.length);

            /* Write the checksum for the data into the header. */
            ChecksumHeader* header = reinterpret_cast<ChecksumHeader*>(iov_elem.addr);
            uint32_t checksum = htonl(static_cast<uint32_t>(data_char) * static_cast<uint32_t>(iov_elem.length));
            m_mem_utils->memory_copy(&header->checksum, &checksum, sizeof(checksum));
        }
    }
}
