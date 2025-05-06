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

#ifndef RMAX_APPS_LIB_LIB_SERVICES_MEMORY_MANAGEMENT_HUGE_PAGES_MEMORY_ALLOCATOR_H_
#define RMAX_APPS_LIB_LIB_SERVICES_MEMORY_MANAGEMENT_HUGE_PAGES_MEMORY_ALLOCATOR_H_

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
 * @brief: Huge Page memory utilities.
 *
 * Implements @ref ral::lib::services::MemoryUtils interface.
 */
class HugePageMemoryUtils : public MemoryUtils
{
};
/**
 * @brief: Huge Pages memory allocation.
 *
 * Implements @ref ral::lib::services::MemoryAllocator interface for allocating memory using Huge Pages.
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
     * @param [in] page_size_log2 - log2 of selected Huge Page size. HUGE_PAGE_SIZE_VALUE_AUTO - default
     */
    HugePagesMemoryAllocator(int page_size_log2);
    ~HugePagesMemoryAllocator();
    void* allocate(const size_t length) override;
    std::shared_ptr<MemoryUtils> get_memory_utils() override;
    size_t get_page_size() const override { return m_page_size; }
};

} // namespace services
} // namespace lib
} // namespace ral

#endif /* RMAX_APPS_LIB_LIB_SERVICES_MEMORY_MANAGEMENT_HUGE_PAGES_MEMORY_ALLOCATOR_H_ */
