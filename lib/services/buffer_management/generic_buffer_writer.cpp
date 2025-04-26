/*
 * Copyright © 2017-2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
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
    auto packets = app_chunk.get_packets_ptr();

    for (size_t packet_index = 0; packet_index < app_chunk.get_length(); packet_index++) {
        auto& packet = packets[packet_index];
        for (size_t iovec_index = 0; iovec_index < packet.count; iovec_index++) {
            auto& iovec = packet.iovec[iovec_index];
            mem_utils->memory_set(reinterpret_cast<void*>(iovec.addr), '0' + (packet_index % 9), iovec.length);
        }
    }
}
