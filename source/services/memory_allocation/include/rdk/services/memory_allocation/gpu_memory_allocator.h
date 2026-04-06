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

#ifndef RDK_SERVICES_MEMORY_ALLOCATION_GPU_MEMORY_ALLOCATOR_H_
#define RDK_SERVICES_MEMORY_ALLOCATION_GPU_MEMORY_ALLOCATOR_H_

#include <cstddef>
#include <memory>
#include <functional>

#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/memory_allocation/memory_allocator_interface.h"

struct gpu_stream;

namespace rdk
{
namespace services
{

/**
 * @brief: GPU memory utilities.
 *
 * Implements @ref MemoryUtils interface.
 */
class GpuMemoryUtils : public MemoryUtils
{
public:
    GpuMemoryUtils();
    ~GpuMemoryUtils() = default;

    MemoryLocation get_memory_location() const override { return MemoryLocation::GPU; }
    ReturnStatus memory_set(void* dst, int value, size_t count) const override;
    ReturnStatus memory_copy(void* dst, const void* src, size_t count) const override;
    ReturnStatus memory_copy_from(void* dst, const void* src,
        size_t count, MemoryLocation  src_location) const override;
    ReturnStatus memory_copy_to(void* dst, const void* src,
       size_t count, MemoryLocation dst_location) const override;
    ReturnStatus memory_copy_2D(void* dst, size_t dst_padded_width,
        const void* src, size_t src_padded_width, size_t width, size_t height,
        MemoryLocation src_location) const override;
private:
    using stream_deleter_t = std::function<void(gpu_stream*)>;
    mutable std::unique_ptr<gpu_stream, stream_deleter_t> m_stream;
};
/**
 * @brief: GPU memory allocation.
 *
 * Implements @ref MemoryAllocator interface for allocating memory using GPU device.
 */
class GpuMemoryAllocator : public MemoryAllocator
{
public:
    explicit GpuMemoryAllocator(int gpu_id);
    ~GpuMemoryAllocator();
    void* allocate(const size_t length) override;
    std::shared_ptr<MemoryUtils> get_memory_utils() override;
    MemoryLocation get_memory_location() const override { return MemoryLocation::GPU; }
    size_t get_page_size() const override;
private:
    int m_gpu_id;
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_MEMORY_ALLOCATION_GPU_MEMORY_ALLOCATOR_H_ */
