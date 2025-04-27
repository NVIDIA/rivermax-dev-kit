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
