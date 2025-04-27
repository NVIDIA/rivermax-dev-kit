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

#ifndef RMAX_APPS_LIB_LIB_SERVICES_MEMORY_MANAGEMENT_GPU_MEMORY_ALLOCATOR_H_
#define RMAX_APPS_LIB_LIB_SERVICES_MEMORY_MANAGEMENT_GPU_MEMORY_ALLOCATOR_H_

#include <cstddef>

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
