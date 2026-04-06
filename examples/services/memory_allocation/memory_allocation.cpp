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

#include <cstddef>

#include "rdk/facade.h"
#include "rdk/services/utils/defs.h"
#include "rdk/examples/services/memory_allocation/memory_allocation.h"

using namespace rdk;
using namespace rdk::services;
using namespace rdk::examples::services;

namespace rdk
{
namespace examples
{
namespace services
{

constexpr const char* EXAMPLE_DESCRIPTION = "NVIDIA Rivermax Dev Kit Memory Allocation Example";
constexpr const char* EXAMPLE_USAGE = "Examples:\n"
                                      "  1. memory_allocators_example"
#ifdef CUDA_ENABLED
                                      "\n  2. memory_allocators_example --gpu-id 0"
#else
                                      ""
#endif
    ;

MemoryAllocationExample::MemoryAllocationExample() :
    BaseExample(EXAMPLE_DESCRIPTION, EXAMPLE_USAGE)
{
}

void MemoryAllocationExample::add_cli_options()
{
#ifdef CUDA_ENABLED
    m_cli_parser_manager->add_option(CLIOptStr::GPU_ID);
#endif
}

ReturnStatus MemoryAllocationExample::operator()()
{
    size_t allocation_size = 128_MB;

    /** 1. Get facade instance */
    auto& facade = RivermaxDevKitFacade::get_instance();

    /** 2.1 Allocate memory using Hugepage allocator */
    auto hugepage_allocator = facade.get_memory_allocator(AllocatorType::HugePageDefault, m_settings);
    RETURN_STATUS_IF(!hugepage_allocator, "Failed to get hugepage allocator", ReturnStatus::failure);
    void* hugepage_memory_ptr = hugepage_allocator->allocate(allocation_size);
    RETURN_STATUS_IF(!hugepage_memory_ptr, "Failed to allocate memory", ReturnStatus::failure);

    /** 2.2 Allocate memory using GPU allocator */
    if (m_settings->gpu_id != INVALID_GPU_ID) {
        auto gpu_allocator = facade.get_memory_allocator(AllocatorType::GPU, m_settings);
        RETURN_STATUS_IF(!gpu_allocator, "Failed to get GPU allocator", ReturnStatus::failure);
        void* gpu_memory_ptr = gpu_allocator->allocate(allocation_size);
        RETURN_STATUS_IF(!gpu_memory_ptr, "Failed to allocate memory", ReturnStatus::failure);
    }

    return ReturnStatus::success;
}

} // namespace services
} // namespace examples
} // namespace rdk

int main(int argc, const char* argv[])
{
    return rdk::examples::common_example_main<MemoryAllocationExample>(argc, argv);
}
