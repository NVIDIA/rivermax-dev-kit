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

#ifndef RDK_IO_NODE_COMMON_CHUNK_BUFFER_WRITER_INTERFACE_H_
#define RDK_IO_NODE_COMMON_CHUNK_BUFFER_WRITER_INTERFACE_H_

#include <memory>

#include "rdk/services/memory_allocation/memory_allocation.h"
#include "rdk/core/chunk/chunk.h"

using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
{
namespace io_node
{

/**
 * @brief: Data buffers writer interface.
 */
class IChunkBufferWriter
{
public:
    /**
     * @brief: IChunkBufferWriter constructor.
     */
    IChunkBufferWriter(std::shared_ptr<MemoryUtils> mem_utils) : m_mem_utils(std::move(mem_utils)) {}
    virtual ~IChunkBufferWriter() = default;
    /**
     * @brief: Writes data buffers.
     *
     * Write the data to the given chunk.
     *
     * @param [in] chunk: Chunk reference.
     */
    virtual void write_buffer(IChunk& chunk) = 0;
protected:
    std::shared_ptr<MemoryUtils> m_mem_utils;
};

} // namespace io_node
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_IO_NODE_COMMON_CHUNK_BUFFER_WRITER_INTERFACE_H_ */
