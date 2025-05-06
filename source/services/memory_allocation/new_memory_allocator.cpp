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
#include "rdk/services/memory_allocation/new_memory_allocator.h"

using namespace rivermax::dev_kit::services;

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
