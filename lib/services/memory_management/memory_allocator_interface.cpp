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

#include <iostream>

#ifdef __linux__
#include <unistd.h>
#else // __linux__
#include <sysinfoapi.h>
#endif // __linux__

#include "gpu.h"

#include "services/memory_management/memory_allocator_interface.h"
#include "services/memory_management/new_memory_allocator.h"
#include "services/memory_management/gpu_memory_allocator.h"
#include "services/error_handling/return_status.h"
#include "services/utils/defs.h"
#include "services/cli/cli.h"

using namespace ral::lib::services;

mem_allocator_factory_map_t MemoryAllocator::s_mem_allocator_factory = \
{
    {
        AllocatorType::New,
        [](std::shared_ptr<AppSettings> app_settings)
        {
            NOT_IN_USE(app_settings);
            return std::shared_ptr<MemoryAllocator>(new NewMemoryAllocator);
        }
    },
    {
        AllocatorType::Gpu,
        [](std::shared_ptr<AppSettings> app_settings)
        {
            if (app_settings->gpu_id != INVALID_GPU_ID) {
                return std::shared_ptr<MemoryAllocator>(new GpuMemoryAllocator(app_settings->gpu_id));
            } else {
                return std::shared_ptr<MemoryAllocator>(nullptr);
            }
            }
     },
};

void* MemoryAllocatorImp::allocate_new(const size_t length)
{
    byte_t* mem_ptr = new (std::nothrow) byte_t[length];
    if (mem_ptr == nullptr) {
        std::cerr << "Failed to allocate " << length << " bytes" << std::endl;
        return nullptr;
    }

    return mem_ptr;
}

ReturnStatus MemoryAllocatorImp::free_new(void* mem_ptr)
{
    if (mem_ptr == nullptr) {
        std::cerr << "Failed to free the pointer at address " << mem_ptr << std::endl;
        return ReturnStatus::failure;
    }

    delete[] static_cast<byte_t*>(mem_ptr);
    mem_ptr = nullptr;

    return ReturnStatus::success;
}

std::shared_ptr<MemoryUtils> MemoryAllocatorImp::get_memory_utils_new()
{
    if (!utils_new) {
        utils_new.reset(new NewMemoryUtils);
    }
    return utils_new;
}

void* MemoryAllocatorImp::allocate_gpu(int gpu_id, size_t length)
{
    return gpu_allocate_memory(gpu_id, length, 0);
}

ReturnStatus MemoryAllocatorImp::free_gpu(void* mem_ptr, size_t length)
{
    return gpu_free_memory(mem_ptr, length) ? ReturnStatus::success : ReturnStatus::failure;
}

std::shared_ptr<MemoryUtils> MemoryAllocatorImp::get_memory_utils_gpu()
{
    if (!utils_gpu) {
        utils_gpu.reset(new GpuMemoryUtils);
    }
    return utils_gpu;
}

#ifdef __linux__

/**
 * @brief: Linux memory allocator implementation.
 *
 * Implements @ref ral::lib::services::MemoryAllocatorImp interface for Linux specific memory allocation.
 */
class LinuxMemoryAllocatorImp : public MemoryAllocatorImp
{
public:
    size_t get_os_page_size() const final
    {
        return static_cast<size_t>(sysconf(_SC_PAGE_SIZE));
    }
};

#else // __linux__

/**
 * @brief: Windows memory allocator implementation.
 *
 * Implements @ref ral::lib::services::MemoryAllocatorImp interface for Windows specific memory allocation.
 */
class WindowsMemoryAllocatorImp : public MemoryAllocatorImp
{
    size_t get_os_page_size() const final
    {
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        return static_cast<size_t>(sysInfo.dwPageSize);
    }
};

#endif // __linux__

MemoryAllocator::MemoryAllocator() :
    m_imp(get_os_imp())
{
}

void* MemoryAllocator::allocate_aligned(size_t length, size_t alignment)
{
    byte_ptr_t ptr = static_cast<byte_ptr_t>(allocate(length + alignment));
    size_t remainder = reinterpret_cast<size_t>(ptr) % alignment;
    if (remainder == 0) {
        return ptr;
    } else {
        return ptr + (alignment - remainder);
    }
}

size_t MemoryAllocator::align_length(size_t length)
{
    return (length + get_page_size() - 1) / get_page_size() * get_page_size();
}

std::unique_ptr<MemoryAllocatorImp> MemoryAllocator::get_os_imp()
{
#ifdef __linux__
    return std::unique_ptr<MemoryAllocatorImp>(new LinuxMemoryAllocatorImp());
#elif _WIN32
    return std::unique_ptr<MemoryAllocatorImp>(new WindowsMemoryAllocatorImp());
#else
    throw "This OS is not yet implemented";
#endif
}

std::shared_ptr<MemoryAllocator> MemoryAllocator::get_memory_allocator(
    AllocatorType type, std::shared_ptr<AppSettings> app_settings)
{
    auto iter = MemoryAllocator::s_mem_allocator_factory.find(type);
    if (iter != MemoryAllocator::s_mem_allocator_factory.end()) {
        return iter->second(app_settings);
    }
    else {
        return nullptr;
    }
}
