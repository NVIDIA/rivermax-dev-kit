/*
 * Copyright © 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#include <iostream>
#include <algorithm>

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
    throw std::logic_error("Not implemented");
}
