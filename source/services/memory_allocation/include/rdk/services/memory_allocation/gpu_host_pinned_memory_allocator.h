/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef RDK_SERVICES_MEMORY_ALLOCATION_GPU_HOST_PINNED_MEMORY_ALLOCATOR_H_
#define RDK_SERVICES_MEMORY_ALLOCATION_GPU_HOST_PINNED_MEMORY_ALLOCATOR_H_

#include <memory>

#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/memory_allocation/memory_allocator_interface.h"

namespace rdk
{
namespace services
{

/**
 * @brief: GPU Host Pinned memory utilities.
 *
 * Implements @ref MemoryUtils interface.
 */
class GpuHostPinnedMemoryUtils : public MemoryUtils
{
public:
    MemoryLocation get_memory_location() const override { return MemoryLocation::Host; }
};

/**
 * @brief: GPU Host Pinned memory allocation.
 *
 * Implements @ref MemoryAllocator interface for allocating GPU host pinned memory.
 */
class GpuHostPinnedMemoryAllocator : public MemoryAllocator {
private:
    const size_t m_page_size;
public:
    /**
     * @brief: Initializes GPU Host Pinned memory allocator.
     */
    GpuHostPinnedMemoryAllocator();
    ~GpuHostPinnedMemoryAllocator();
    void* allocate(const size_t length) override;
    std::shared_ptr<MemoryUtils> get_memory_utils() override;
    MemoryLocation get_memory_location() const override { return MemoryLocation::Host; }
    size_t get_page_size() const override { return m_page_size; }
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_MEMORY_ALLOCATION_GPU_HOST_PINNED_MEMORY_ALLOCATOR_H_ */
