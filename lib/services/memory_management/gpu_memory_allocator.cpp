/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <iostream>
#include <algorithm>
#include <cstring>

#include "gpu.h"

#include "services/memory_management/gpu_memory_allocator.h"
#include "services/error_handling/return_status.h"

using namespace ral::lib::services;

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
    if (!gpu_init(gpu_id)) {
        throw std::runtime_error("Failed to init GPU device");
    }

    int res = gpu_set_locked_clocks_max_freq(gpu_id);
    bool set_freq_caused_failure = (res != 0) && (res != -ENOTSUP);
    if (set_freq_caused_failure) {
        throw std::runtime_error("Failed to set GPU clock!");
    }

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

    gpu_reset_locked_clocks(m_gpu_id);

    if (!gpu_uninit(m_gpu_id)) {
        std::cerr << "Failed to uninit GPU device" << std::endl;
    }
}

std::shared_ptr<MemoryUtils> GpuMemoryAllocator::get_memory_utils()
{
    return m_imp->get_memory_utils_gpu();
}

size_t GpuMemoryAllocator::get_page_size() const
{
    return gpu_query_alignment(m_gpu_id);
}
