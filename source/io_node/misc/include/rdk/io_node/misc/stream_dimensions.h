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

#ifndef RDK_IO_NODE_MISC_STREAM_DIMENSIONS_H_
#define RDK_IO_NODE_MISC_STREAM_DIMENSIONS_H_

#include <cstddef>

namespace rdk
{
namespace io_node
{

/**
 * @brief: Data dimensions of a generic stream.
 */
struct StreamDimensions
{
    /* Number of chunks in the stream */
    std::size_t num_of_chunks = 0;
    /* Number of packets in a single chunk */
    std::size_t num_of_packets_in_chunk = 0;
    /* Size of a packet header in bytes, 0 when header-data-split is not used */
    std::size_t header_size = 0;
    /* Size of a packet payload in bytes */
    std::size_t payload_size = 0;

    StreamDimensions() = default;
    /**
     * @brief: StreamDimensions constructor.
     *
     * @param [in] a_num_of_chunks: Number of chunks in the stream.
     * @param [in] a_num_of_packets: Number of packets in a single chunk.
     * @param [in] a_header_size: Size of a packet header in bytes.
     * @param [in] a_payload_size: Size of a packet payload in bytes.
     */
    StreamDimensions(std::size_t a_num_of_chunks, std::size_t a_num_of_packets,
                     std::size_t a_header_size, std::size_t a_payload_size) :
        num_of_chunks(a_num_of_chunks),
        num_of_packets_in_chunk(a_num_of_packets),
        header_size(a_header_size),
        payload_size(a_payload_size) {}
};

} // namespace io_node
} // namespace rdk

#endif /* RDK_IO_NODE_MISC_STREAM_DIMENSIONS_H_ */
