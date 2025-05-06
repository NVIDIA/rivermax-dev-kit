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

#include <iostream>
#include <algorithm>
#include <cstring>

#include "gpu.h"

#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/memory_allocation/gpu_memory_allocator.h"

using namespace rivermax::dev_kit::services;

std::shared_ptr<MemoryUtils> MemoryAllocatorImp::utils_gpu;

ReturnStatus GpuMemoryUtils::memory_set(void* dst, int value, size_t count) const
{
    return gpu_memset(dst, value, count) ? ReturnStatus::success : ReturnStatus::failure;
}

ReturnStatus GpuMemoryUtils::memory_copy(void* dst, const void* src, size_t count) const
{
    return gpu_memcpy(dst, src, count) ? ReturnStatus::success : ReturnStatus::failure;
}

GpuMemoryAllocator::GpuMemoryAllocator(int gpu_id)
    : MemoryAllocator()
    , m_gpu_id(gpu_id)
{
    if (!set_gpu_device(m_gpu_id)) {
        throw std::runtime_error("Failed to set GPU device!");
    }
}

void* GpuMemoryAllocator::allocate(const size_t length)
{
    void* mem_ptr = m_imp->allocate_gpu(m_gpu_id, length);
    if (!mem_ptr) {
        std::cerr << "Failed to allocate memory on GPU id " << m_gpu_id << std::endl;
        return nullptr;
    }

    m_mem_blocks.push_back(std::unique_ptr<mem_block_t>(new mem_block_t{ mem_ptr, length }));

    return mem_ptr;
}

GpuMemoryAllocator::~GpuMemoryAllocator()
{
    std::for_each(m_mem_blocks.begin()
                , m_mem_blocks.end()
                , [this](std::unique_ptr<mem_block_t>& mem_block){ m_imp->free_gpu(mem_block->pointer, mem_block->length); });
}

std::shared_ptr<MemoryUtils> GpuMemoryAllocator::get_memory_utils()
{
    return m_imp->get_memory_utils_gpu();
}

size_t GpuMemoryAllocator::get_page_size() const
{
    return gpu_query_alignment(m_gpu_id);
}
