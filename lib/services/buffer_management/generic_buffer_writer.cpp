/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cstddef>

#include "core/chunk/chunk_interface.h"
#include "core/chunk/generic_chunk.h"
#include "services/buffer_management/generic_buffer_writer.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

void GenericBufferWriter::write_buffer(IChunk& chunk, std::shared_ptr<MemoryUtils> mem_utils)
{
    auto& app_chunk = dynamic_cast<GenericChunk&>(chunk);

    for (size_t packet_index = 0; packet_index < app_chunk.get_length(); packet_index++) {
        auto& packet = app_chunk.get_packet(packet_index);
        for (auto& iov_elem : packet) {
            mem_utils->memory_set(reinterpret_cast<void*>(iov_elem.addr), '0' + (packet_index % 9), iov_elem.length);
        }
    }
}
