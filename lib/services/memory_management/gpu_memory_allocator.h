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

#ifndef RMAX_APPS_LIB_LIB_SERVICES_MEMORY_MANAGEMENT_GPU_MEMORY_ALLOCATOR_H_
#define RMAX_APPS_LIB_LIB_SERVICES_MEMORY_MANAGEMENT_GPU_MEMORY_ALLOCATOR_H_

#include <cstddef>
#include <memory>

#include "services/memory_management/memory_allocator_interface.h"
#include "services/error_handling/return_status.h"

namespace ral
{
namespace lib
{
namespace services
{

/**
 * @brief: GPU memory utilities.
 *
 * Implements @ref ral::lib::services::MemoryUtils interface.
 */
class GpuMemoryUtils : public MemoryUtils
{
public:
    ReturnStatus memory_set(void* dst, int value, size_t count) const override;
    ReturnStatus memory_copy(void* dst, const void* src, size_t count) const override;
};
/**
 * @brief: GPU memory allocation.
 *
 * Implements @ref ral::lib::services::MemoryAllocator interface for allocating memory using GPU device.
 */
class GpuMemoryAllocator : public MemoryAllocator
{
public:
    explicit GpuMemoryAllocator(int gpu_id);
    ~GpuMemoryAllocator();
    void* allocate(const size_t length) override;
    std::shared_ptr<MemoryUtils> get_memory_utils() override;
    size_t get_page_size() const override;
private:
    int m_gpu_id;
};

} // namespace services
} // namespace lib
} // namespace ral

#endif /* RMAX_APPS_LIB_LIB_SERVICES_MEMORY_MANAGEMENT_GPU_MEMORY_ALLOCATOR_H_ */
