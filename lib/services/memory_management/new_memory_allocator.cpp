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
