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

#ifndef RDK_SERVICES_MEMORY_ALLOCATION_HUGE_PAGES_MEMORY_ALLOCATOR_H_
#define RDK_SERVICES_MEMORY_ALLOCATION_HUGE_PAGES_MEMORY_ALLOCATOR_H_

#include <cstddef>

#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/memory_allocation/memory_allocator_interface.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Huge Page memory utilities.
 *
 * Implements @ref MemoryUtils interface.
 */
class HugePageMemoryUtils : public MemoryUtils
{
};
/**
 * @brief: Huge Pages memory allocation.
 *
 * Implements @ref MemoryAllocator interface for allocating memory using Huge Pages.
 */
class HugePagesMemoryAllocator : public MemoryAllocator
{
private:
    size_t m_page_size;
    size_t align_length(size_t length) final;
public:
    /**
     * @brief: Initializes Huge Page memory allocator.
     *
     * @param [in] page_size_log2: log2 of selected Huge Page size. HUGE_PAGE_SIZE_VALUE_AUTO - default
     */
    HugePagesMemoryAllocator(int page_size_log2);
    ~HugePagesMemoryAllocator();
    void* allocate(const size_t length) override;
    std::shared_ptr<MemoryUtils> get_memory_utils() override;
    size_t get_page_size() const override { return m_page_size; }
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEMORY_ALLOCATION_HUGE_PAGES_MEMORY_ALLOCATOR_H_ */
