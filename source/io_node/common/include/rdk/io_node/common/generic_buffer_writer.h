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

#ifndef RDK_IO_NODE_COMMON_GENERIC_BUFFER_WRITER_H_
#define RDK_IO_NODE_COMMON_GENERIC_BUFFER_WRITER_H_

#include "rdk/io_node/common/chunk_buffer_writer_interface.h"
#include "rdk/core/chunk/chunk.h"

using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
{
namespace io_node
{

/**
 * @brief: Generic data buffers writer.
 *
 * This class implements @ref IChunkBufferWriter interface.
 * It writes the buffer by some internal logic.
 * There is no special reason why this buffer writer writes it's data the way it does.
 */
class GenericBufferWriter : public IChunkBufferWriter
{
public:
    GenericBufferWriter(std::shared_ptr<MemoryUtils> mem_utils) : IChunkBufferWriter(std::move(mem_utils)) {}
    void write_buffer(IChunk& chunk) override;
};

} // namespace io_node
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_IO_NODE_COMMON_GENERIC_BUFFER_WRITER_H_ */
