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

#include "rdk/services/utils/defs.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/memory_allocation/huge_pages_memory_allocator.h"

using namespace rivermax::dev_kit::services;

std::shared_ptr<MemoryUtils> MemoryAllocatorImp::utils_huge_pages;

HugePagesMemoryAllocator::HugePagesMemoryAllocator(int page_size_log2)
{
    if (!m_imp->init_huge_pages(page_size_log2, m_page_size)) {
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
