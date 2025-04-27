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

#include "services/memory_management/new_memory_allocator.h"
#include "services/error_handling/return_status.h"
#include "services/utils/defs.h"

using namespace ral::lib::services;

std::shared_ptr<MemoryUtils> MemoryAllocatorImp::utils_new;

NewMemoryAllocator::NewMemoryAllocator() :
    m_page_size(m_imp->get_os_page_size())
{
}

NewMemoryAllocator::~NewMemoryAllocator()
{
    ReturnStatus rc;
    for (auto& mem_block : m_mem_blocks) {
        rc = m_imp->free_new(mem_block->pointer);
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to free memory using C++ delete[] operator" << std::endl;
        }
    }
}

void* NewMemoryAllocator::allocate(const size_t length)
{
    void* mem_ptr = m_imp->allocate_new(length);
    if (!mem_ptr) {
        std::cerr << "Failed to allocate memory using C++ new operator" << std::endl;
        return nullptr;
    }

    m_mem_blocks.push_back(std::unique_ptr<mem_block_t>(new mem_block_t{ mem_ptr, length }));

    return mem_ptr;
}

std::shared_ptr<MemoryUtils> NewMemoryAllocator::get_memory_utils()
{
    return m_imp->get_memory_utils_new();
}
