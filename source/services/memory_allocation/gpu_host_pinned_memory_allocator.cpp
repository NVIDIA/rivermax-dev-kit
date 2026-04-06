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

#include <iostream>
#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#include "rdk/services/utils/defs.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/memory_allocation/gpu_host_pinned_memory_allocator.h"

using namespace rdk::services;

std::shared_ptr<MemoryUtils> MemoryAllocatorImp::utils_gpu_host_pinned;

GpuHostPinnedMemoryAllocator::GpuHostPinnedMemoryAllocator() :
    m_page_size(m_imp->get_os_page_size())
{
}

GpuHostPinnedMemoryAllocator::~GpuHostPinnedMemoryAllocator()
{
    ReturnStatus rc;
    for (auto& mem_block : m_mem_blocks) {
        rc = m_imp->free_gpu_host_pinned(mem_block->pointer);
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to free GPU host pinned memory" << std::endl;
        }
    }
}

void* GpuHostPinnedMemoryAllocator::allocate(const size_t length)
{
    void* mem_ptr = m_imp->allocate_gpu_host_pinned(length);
    if (!mem_ptr) {
        std::cerr << "Failed to allocate GPU host pinned memory" << std::endl;
        return nullptr;
    }

    m_mem_blocks.push_back(std::unique_ptr<mem_block_t>(new mem_block_t{ mem_ptr, length }));
    return mem_ptr;
}

std::shared_ptr<MemoryUtils> GpuHostPinnedMemoryAllocator::get_memory_utils()
{
    return m_imp->get_memory_utils_gpu_host_pinned();
}
