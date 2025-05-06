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

#ifndef RDK_CORE_MEMORY_LAYOUT_MEDIA_MEMORY_LAYOUT_H_
#define RDK_CORE_MEMORY_LAYOUT_MEDIA_MEMORY_LAYOUT_H_

#include "rdk/core/stream/send/media_stream.h"
#include "rdk/core/memory_layout/memory_layout_interface.h"

namespace rivermax
{
namespace dev_kit
{
namespace core
{
// Forward declaration
class MediaStreamMemBlockset;
/**
 * @brief: Memory layout configuration for media components.
 *
 * This struct holds the memory layout configuration for media streams, including detailed memory layouts.
 */
struct MediaMemoryLayoutResponse : public MemoryLayoutResponse
{
    /**
     * @brief: Constructor for MediaMemoryLayoutResponse.
     *
     * @param [in] mem_blocks_: Reference to the MediaStreamMemBlockset object.
     */
    MediaMemoryLayoutResponse(MediaStreamMemBlockset& mem_blocks_) : mem_blocks(mem_blocks_) {}
    /**
     * @brief: Destructor for MediaMemoryLayoutResponse.
     */
    virtual ~MediaMemoryLayoutResponse() = default;
    /* Detailed memory layouts for streams. */
    MediaStreamMemBlockset& mem_blocks;
};

} // namespace core
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_CORE_MEMORY_LAYOUT_MEDIA_MEMORY_LAYOUT_H_ */
