/*
 * Copyright © 2017-2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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

#include "services/memory_management/huge_pages_memory_allocator.h"
#include "services/error_handling/return_status.h"
#include "services/utils/defs.h"

using namespace ral::lib::services;

std::shared_ptr<MemoryUtils> MemoryAllocatorImp::utils_huge_pages;

HugePagesMemoryAllocator::HugePagesMemoryAllocator()
{
    if (!m_imp->init_huge_pages(m_page_size)) {
        std::cerr << "Failed to initialize Huge Pages" << std::endl;
    }
}

HugePagesMemoryAllocator::~HugePagesMemoryAllocator()
{
    ReturnStatus rc;
    for (auto& mem_block : m_mem_blocks) {
        rc = m_imp->free_huge_pages(mem_block->pointer, mem_block->length);
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to free Huge Pages memory" << std::endl;
        }
    }
}

void* HugePagesMemoryAllocator::allocate(const size_t length)
{
    size_t aligned_length = align_length(length);
    void* mem_ptr = m_imp->allocate_huge_pages(aligned_length, m_page_size);
    if (!mem_ptr) {
        std::cerr << "Failed to allocate memory using Huge Pages" << std::endl;
        return nullptr;
    }

    m_mem_blocks.push_back(std::unique_ptr<mem_block_t>(new mem_block_t{ mem_ptr, aligned_length }));

    return mem_ptr;
}

size_t HugePagesMemoryAllocator::align_length(size_t length)
{
    size_t factor = length / m_page_size;
    factor += (length % m_page_size > 0) ? 1 : 0;
    return factor * m_page_size;
}

std::shared_ptr<MemoryUtils> HugePagesMemoryAllocator::get_memory_utils()
{
    return m_imp->get_memory_utils_huge_pages();
}
