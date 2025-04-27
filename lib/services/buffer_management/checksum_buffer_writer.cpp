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
#include "services/buffer_management/checksum_buffer_writer.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

void ChecksumBufferWriter::write_buffer(IChunk& chunk, std::shared_ptr<MemoryUtils> mem_utils)
{
    auto& app_chunk = static_cast<GenericChunk&>(chunk);

    for (size_t packet_index = 0; packet_index < app_chunk.get_length(); packet_index++) {
        auto& packet = app_chunk.get_packet(packet_index);
        for (auto& iov_elem : packet) {
            /* Fill the data with a randomly repeating char. */
            unsigned char data_char = rand() % 255;
            mem_utils->memory_set(reinterpret_cast<void*>(
                reinterpret_cast<uint8_t*>(iov_elem.addr) + iov_elem.length), data_char, iov_elem.length);

            /* Write the checksum for the data into the header. */
            ChecksumHeader* header = reinterpret_cast<ChecksumHeader*>(iov_elem.addr);
            uint32_t checksum = htonl(static_cast<uint32_t>(data_char) * static_cast<uint32_t>(iov_elem.length));
            mem_utils->memory_copy(&header->checksum, &checksum, sizeof(checksum));
        }
    }
}
