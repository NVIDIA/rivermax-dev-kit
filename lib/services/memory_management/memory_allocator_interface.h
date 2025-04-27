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

#ifndef RMAX_APPS_LIB_LIB_SERVICES_MEMORY_MANAGEMENT_MEMORY_ALLOCATOR_INTERFACE_H_
#define RMAX_APPS_LIB_LIB_SERVICES_MEMORY_MANAGEMENT_MEMORY_ALLOCATOR_INTERFACE_H_

#include <cstddef>
#include <memory>
#include <vector>

#include <rivermax_api.h>

#include "services/error_handling/return_status.h"
#include "services/cli/cli.h"

namespace ral
{
namespace lib
{
namespace services
{
/**
 * @brief: Huge Page allocator constants.
 */
static constexpr int HUGE_PAGE_SIZE_VALUE_AUTO = 0;
static constexpr int HUGE_PAGE_SIZE_VALUE_2MB = 21;
static constexpr int HUGE_PAGE_SIZE_VALUE_512MB = 29;
static constexpr int HUGE_PAGE_SIZE_VALUE_1GB = 30;
/**
 * @brief: Allocator types supported.
 */
enum class AllocatorType {
    Malloc,
    HugePageDefault,
    HugePage2MB,
    HugePage512MB,
    HugePage1GB,
    Gpu
};
/**
 * @brief: Memory block representation.
 *
 * The struct represents memory block.
 *
 * @param [in] pointer: Pointer to the start address of the memory.
 * @param [in] length: Memory block length.
 */
typedef struct mem_block
{
    void* pointer;
    size_t length;
} mem_block_t;
/**
 * @brief: Generic API memory block.
 *
 * The struct represents memory block for generic stream.
 *
 * @param [in] mem_block: Memory block.
 * @param [in] mkey_id: Memory block key.
 */
typedef struct gs_mem_block
{
    mem_block_t mem_block;
    rmx_mkey_id mkey_id;
} gs_mem_block_t;
/**
 * @brief: Memory utils interface.
 *
 * The memory utils interface should be implemented when adding new type of @ref ral::lib::services:MemoryAllocator
 * that need new memory utilities to handle the new memory type.
 * Implementors of this interface, should add implementation of compatible methods
 * in @ref ral::lib::services::MemoryAllocatorImp class.
 */
class MemoryUtils
{
public:
    /**
     * @brief: MemoryUtils default constructor.
     */
    MemoryUtils() {};
    virtual ~MemoryUtils() {};
    /**
     * @brief: Sets memory to the specified value.
     *
     * @param [in] dst: Destination memory address.
     * @param [in] value: Value to set for each byte of specified memory.
     * @param [in] count: Size in bytes to set.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus memory_set(void* dst, int value, size_t count) const;
    /**
     * @brief: Copies memory.
     *
     * @param [in] dst: Destination memory address.
     * @param [in] src:  Source memory address.
     * @param [in] count: Size in bytes to copy.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus memory_copy(void* dst, const void* src, size_t count) const;
};
/**
 * @brief: Memory allocator implementation.
 *
 * The base class for cross platform memory allocation and deallocation.
 *
 * All future memory allocation methods, e.g. huge pages allocation,
 * should be declared here as virtual methods and implemented in the derived OS specific classes.
 * Currently implemented memory allocation using C++ new and delete operator.
 */
class MemoryAllocatorImp
{
private:
    static std::shared_ptr<MemoryUtils> utils_new;
    static std::shared_ptr<MemoryUtils> utils_huge_pages;
    static std::shared_ptr<MemoryUtils> utils_gpu;
public:
    MemoryAllocatorImp() = default;
    virtual ~MemoryAllocatorImp() = default;
    /**
     * @brief: Allocates memory using C++ new operator.
     *
     * @param [in] length: Length of the memory to allocate.
     *
     * @return: Pointer to the allocated memory.
     */
    virtual void* allocate_new(const size_t length);
    /**
     * @brief: Frees memory using C++ delete operator.
     *
     * @param [in] mem_ptr: Pointer to the memory to free.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus free_new(void* mem_ptr);
    /**
     * @brief: Returns new memory utils.
     *
     * @return: Shared pointer to the memory utils.
     */
     virtual std::shared_ptr<MemoryUtils> get_memory_utils_new();
    /**
     * @brief: Initializes huge pages.
     *
     * @param [in] page_size_log2: log2 of selected Huge Page size. HUGE_PAGE_SIZE_VALUE_AUTO - default
     * @param [out] huge_page_size: Supported Huge Page size.
     *
     * @return: Return true in success, false otherwise.
     */
    virtual bool init_huge_pages(int page_size_log2, size_t& huge_page_size) = 0;
    /**
     * @brief: Allocates memory using Huge Pages alocation.
     *
     * @param [in] length   : Length of the memory to allocate.
     * @param [in] alignment: Aligment size of the memory to allocate.
     *
     * @return: Pointer to the allocated memory.
     */
    virtual void* allocate_huge_pages(size_t length, size_t alignment) = 0;
    /**
     * @brief: Frees memory using Huge Pages delete operator.
     *
     * @param [in] mem_ptr: Pointer to the memory to free.
     * @param [in] length : Length of the memory to free.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus free_huge_pages(void* mem_ptr, size_t length) = 0;
    /**
     * @brief: Return Huge Pages memory utils.
     *
     * @return: Shared pointer to the memory utils.
     */
    virtual std::shared_ptr<MemoryUtils> get_memory_utils_huge_pages();
    /**
     * @brief: Allocates memory using CUDA.
     *
     * @param [in] length: Length of the memory to allocate.
     * @param [in] gpu_id: GPU ID.
     *
     * @return: Pointer to the allocated memory.
     */
    virtual void* allocate_gpu(int gpu_id, size_t length);
    /**
     * @brief: Frees memory using CUDA.
     *
     * @param [in] mem_ptr: Pointer to the memory to free.
     * @param [in] length : Length of the memory to free.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus free_gpu(void* mem_ptr, size_t length);
    /**
     * @brief: Returns GPU memory utils.
     *
     * @return: Shared pointer to the memory utils.
     */
    virtual std::shared_ptr<MemoryUtils> get_memory_utils_gpu();
    /**
     * @brief: Returns operating system's memory page size.
     *
     * @return: Page size in bytes.
     */
    virtual size_t get_os_page_size() const = 0;
};

class MemoryAllocator;
typedef std::unordered_map<
    AllocatorType,
    std::function<std::shared_ptr<MemoryAllocator>(std::shared_ptr<AppSettings>)>,
    EnumClassHash> mem_allocator_factory_map_t;

/**
 * @brief: Memory allocator manager interface.
 *
 * The memory allocator interface to implement when adding new type of
 * memory allocations.
 * Implementors of this interface, should add the memory implementation
 * specific logic, by expanding @ref ral::lib::services::MemoryAllocatorImp interface with
 * a compatible methods for allocation and free of the new added memory allocation type.
 */
class MemoryAllocator
{
protected:
    std::unique_ptr<MemoryAllocatorImp> m_imp;
    std::vector<std::unique_ptr<mem_block_t>> m_mem_blocks;
private:
    static mem_allocator_factory_map_t s_mem_allocator_factory;

public:
    MemoryAllocator();
    virtual ~MemoryAllocator() = default;
    /**
     * @brief: Allocates memory.
     *
     * The method to implement for @ref ral::lib::services::ImpMemoryAllocator interface.
     * Implementors of this interface should delegate the implementation
     * to the compatible method in @ref ral::lib::services::MemoryAllocatorImp implementation.
     *
     * @param [in] length: Length of the memory to allocate.
     *
     * @return: Pointer to the allocated memory.
     */
    virtual void* allocate(const size_t length) = 0;
    /**
     * @brief: Allocates memory and align it to page size.
     *
     * @param [in] length: Requested allocation length.
     * @param [in] alignment: Memory alignment.
     *
     * @return: Pointer to the allocated memory.
     */
    virtual void* allocate_aligned(size_t length, size_t align);
    /**
     * @brief: Returns memory utils.
     *
     * The method to implement for @ref ral::lib::services::MemoryAllocator interface.
     * Implementors of this interface should delegate the implementation
     * to the compatible method in @ref ral::lib::services::MemoryAllocatorImp implementation.
     *
     * @return: Shared pointer to the memory utils.
     */
    virtual std::shared_ptr<MemoryUtils> get_memory_utils() = 0;
    /**
     * @brief: Factory method for memory allocator.
     *
     * @param [in] type: Memory allocator type.
     * @param [in] app_settings: Application settings.
     *
     * @return: Shared pointer to the memory allocator.
     */
    static std::shared_ptr<MemoryAllocator> get_memory_allocator(
        AllocatorType type, std::shared_ptr<AppSettings> app_settings);
    /**
     * @brief: Returns memory page size.
     *
     * @return: Page size in bytes.
     */
    virtual size_t get_page_size() const = 0;
    /**
     * @brief: Rounds memory length up to page size.
     *
     * @return: Rounded value.
     */
    virtual size_t align_length(size_t length);
private:
    /**
     * @brief: Returns OS MemoryAllocatorImp.
     *
     * Factory method to create the OS specific memory allocator implementation object.
     *
     * @return: Returns OS MemoryAllocatorImp unique pointer.
     */
    std::unique_ptr<MemoryAllocatorImp> get_os_imp();
};

} // namespace services
} // namespace lib
} // namespace ral

#endif /* RMAX_APPS_LIB_LIB_SERVICES_MEMORY_MANAGEMENT_MEMORY_ALLOCATOR_INTERFACE_H_ */
