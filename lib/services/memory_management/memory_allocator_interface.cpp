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

#include <cstring>
#include <iostream>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#else // __linux__
#include <sysinfoapi.h>
#pragma comment(lib, "mincore")
#endif // __linux__

#include "gpu.h"

#include "services/memory_management/memory_allocator_interface.h"
#include "services/memory_management/new_memory_allocator.h"
#include "services/memory_management/huge_pages_memory_allocator.h"
#include "services/memory_management/gpu_memory_allocator.h"
#include "services/error_handling/return_status.h"
#include "services/utils/defs.h"
#include "services/cli/cli.h"

using namespace ral::lib::services;

static constexpr int PAGE_SIZE_64KB = 65536;

mem_allocator_factory_map_t MemoryAllocator::s_mem_allocator_factory = \
{
    {
        AllocatorType::Malloc,
        [](std::shared_ptr<AppSettings> app_settings)
        {
            NOT_IN_USE(app_settings);
            return std::shared_ptr<MemoryAllocator>(new NewMemoryAllocator);
        }
    },
    {
        AllocatorType::HugePageDefault,
        [](std::shared_ptr<AppSettings> app_settings)
        {
            NOT_IN_USE(app_settings);
            return std::shared_ptr<MemoryAllocator>(new HugePagesMemoryAllocator(HUGE_PAGE_SIZE_VALUE_AUTO));
        }
    },
    {
        AllocatorType::HugePage2MB,
        [](std::shared_ptr<AppSettings> app_settings)
        {
            NOT_IN_USE(app_settings);
            return std::shared_ptr<MemoryAllocator>(new HugePagesMemoryAllocator(HUGE_PAGE_SIZE_VALUE_2MB));
        }
    },
    {
        AllocatorType::HugePage512MB,
        [](std::shared_ptr<AppSettings> app_settings)
        {
            NOT_IN_USE(app_settings);
            return std::shared_ptr<MemoryAllocator>(new HugePagesMemoryAllocator(HUGE_PAGE_SIZE_VALUE_512MB));
        }
    },
    {
        AllocatorType::HugePage1GB,
        [](std::shared_ptr<AppSettings> app_settings)
        {
            NOT_IN_USE(app_settings);
            return std::shared_ptr<MemoryAllocator>(new HugePagesMemoryAllocator(HUGE_PAGE_SIZE_VALUE_1GB));
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

ReturnStatus MemoryUtils::memory_set(void* dst, int value, size_t count) const
{
    memset(dst, value, count);
    return ReturnStatus::success;
}

ReturnStatus MemoryUtils::memory_copy(void* dst, const void* src, size_t count) const
{
    memcpy(dst, src, count);
    return ReturnStatus::success;
}

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

std::shared_ptr<MemoryUtils> MemoryAllocatorImp::get_memory_utils_huge_pages()
{
    if (!utils_huge_pages) {
        utils_huge_pages.reset(new HugePageMemoryUtils);
    }
    return utils_huge_pages;
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
private:
    int m_huge_page_size_log2 = 0;
public:
    size_t get_os_page_size() const final;
    bool init_huge_pages(int huge_page_size_log2, size_t& huge_page_size) final;
    void* allocate_huge_pages(size_t length, size_t alignment) final;
    ReturnStatus free_huge_pages(void* mem_ptr, size_t length) final;
};

size_t LinuxMemoryAllocatorImp::get_os_page_size() const
{
    return static_cast<size_t>(sysconf(_SC_PAGE_SIZE));
}

bool LinuxMemoryAllocatorImp::init_huge_pages(int huge_page_size_log2, size_t& huge_page_size)
{
    if (huge_page_size_log2 == HUGE_PAGE_SIZE_VALUE_AUTO) {
        switch (sysconf(_SC_PAGESIZE)) {
        case PAGE_SIZE_64KB:
            m_huge_page_size_log2 = HUGE_PAGE_SIZE_VALUE_512MB;
            break;
        default:
            m_huge_page_size_log2 = HUGE_PAGE_SIZE_VALUE_2MB;
        }
    } else {
        m_huge_page_size_log2 = huge_page_size_log2;
    }
    huge_page_size = size_t(1) << m_huge_page_size_log2;
    std::cout << "Init huge pages with size " << huge_page_size << std::endl;
    return true;
}

void* LinuxMemoryAllocatorImp::allocate_huge_pages(size_t length, size_t alignment)
{
    NOT_IN_USE(alignment);
    void* mem_ptr = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGETLB | (m_huge_page_size_log2 << MAP_HUGE_SHIFT), -1, 0);
    if (mem_ptr == MAP_FAILED) {
        std::cerr << "Failed to allocate " << length << " bytes using huge pages with errno: " << errno << ", page size log2: " << m_huge_page_size_log2 << std::endl;
        return nullptr;
    }
    return mem_ptr;
}

ReturnStatus LinuxMemoryAllocatorImp::free_huge_pages(void* mem_ptr, size_t length)
{
    if (mem_ptr == nullptr) {
        std::cerr << "Failed to free the pointer at address " << mem_ptr << std::endl;
        return ReturnStatus::failure;
    }
    if (munmap(mem_ptr, length)) {
        std::cerr << "Failed to free the pointer errno: " << errno << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

#else // __linux__

/**
 * @brief: Windows memory allocator implementation.
 *
 * Implements @ref ral::lib::services::MemoryAllocatorImp interface for Windows specific memory allocation.
 */
class WindowsMemoryAllocatorImp : public MemoryAllocatorImp
{
private:
    uint64_t m_huge_page_extended_flag;
public:
    size_t get_os_page_size() const final;
    bool init_huge_pages(int huge_page_size_log2, size_t& huge_page_size) final;
    void* allocate_huge_pages(size_t length, size_t alignment) final;
    ReturnStatus free_huge_pages(void* mem_ptr, size_t length) final;
};

size_t WindowsMemoryAllocatorImp::get_os_page_size() const
{
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return static_cast<size_t>(sysInfo.dwPageSize);
}

bool WindowsMemoryAllocatorImp::init_huge_pages(int huge_page_size_log2, size_t& huge_page_size)
{
    HANDLE hToken;
    TOKEN_PRIVILEGES tp;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken)) {
        std::cout << "Cannot use Large Pages, could not get privileges" << std::endl;
        return false;
    }

    // Used by local system to identify the privilege
    LUID luid;
    if (!LookupPrivilegeValue(NULL, TEXT("SeLockMemoryPrivilege"), &luid)) {
        std::cout << "Cannot use Large Pages, could not lookup privileges: SeLockMemoryPrivilege" << std::endl;
        CloseHandle(hToken);
        return false;
    }
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    tp.Privileges[0].Luid = luid;
    AdjustTokenPrivileges(hToken, FALSE, &tp, sizeof(TOKEN_PRIVILEGES), NULL, NULL);
    DWORD last_error = GetLastError();
    if (last_error != ERROR_SUCCESS) {
        std::cout << "Cannot use Large Pages, failed setting privileges: error 0x" << std::hex << last_error << std::dec << std::endl;
        std::cout << "Following steps should be done due to enable Large Pages:\n"
            << "1. From the Start menu, open Local Security Policy (under Administrative Tools)\n"
            << "2. Under Local Policies\\User Rights Assignment, double click the Lock Pages in Memory setting\n"
            << "3. Click Add User or Group and type your Windows user name\n"
            << "4. Either log off and then log back in or restart your computer" << std::endl;
        CloseHandle(hToken);
        return false;
    }
    CloseHandle(hToken);

    huge_page_size = GetLargePageMinimum();
    if (huge_page_size == 0) {
        std::cout << "GetLargePageMinimum() error got zero 0x" << std::hex << GetLastError() << std::dec << std::endl;
        return false;
    } else {
        switch (huge_page_size_log2) {
        case HUGE_PAGE_SIZE_VALUE_AUTO:
            m_huge_page_extended_flag = MemExtendedParameterInvalidType;
            break;
        case HUGE_PAGE_SIZE_VALUE_2MB:
            m_huge_page_extended_flag = MEM_EXTENDED_PARAMETER_NONPAGED_LARGE;
            huge_page_size = size_t(1) << huge_page_size_log2;
            break;
        case HUGE_PAGE_SIZE_VALUE_1GB:
            m_huge_page_extended_flag = MEM_EXTENDED_PARAMETER_NONPAGED_HUGE;
            huge_page_size = size_t(1) << huge_page_size_log2;
            break;
        default:
            std::cerr << "Unsupported huge page size log2 value " << huge_page_size_log2 << ". Using system default " << huge_page_size << std::endl;
            m_huge_page_extended_flag = MemExtendedParameterInvalidType;
            break;
        }
    }
    return true;
}

void* WindowsMemoryAllocatorImp::allocate_huge_pages(size_t length, size_t alignment)
{
    NOT_IN_USE(alignment);
    ULONG num_ext_params = 0;
    MEM_EXTENDED_PARAMETER ext_param;
    MEM_EXTENDED_PARAMETER* ext_param_ptr = nullptr;
    if (m_huge_page_extended_flag != MemExtendedParameterInvalidType) {
        num_ext_params = 1;
        ext_param_ptr = &ext_param;
        std::memset(&ext_param, 0, sizeof(ext_param));
        ext_param.Type = MemExtendedParameterAttributeFlags;
        ext_param.ULong64 = m_huge_page_extended_flag;
    }
    void* mem_ptr = VirtualAlloc2(NULL, NULL, length, MEM_LARGE_PAGES | MEM_COMMIT | MEM_RESERVE,
                                  PAGE_READWRITE, ext_param_ptr, num_ext_params);
    if (!mem_ptr) {
        auto last_error = GetLastError();
        std::cerr << "Failed to allocate " << length << " bytes using Large Pages with error: 0x" << std::hex << last_error << std::dec << std::endl;
        if (last_error == ERROR_INVALID_PARAMETER) {
            std::cerr << "Non default large page size is probably not supported. Use default large page size" << std::endl;
        }
        return nullptr;
    }

    std::cout << "Allocated " << length << " bytes using Large Pages" << std::endl;
    return mem_ptr;
}

ReturnStatus WindowsMemoryAllocatorImp::free_huge_pages(void* mem_ptr, size_t length)
{
    NOT_IN_USE(length);
    if (mem_ptr == nullptr) {
        std::cerr << "Failed to free the pointer at address " << mem_ptr << std::endl;
        return ReturnStatus::failure;
    }

    VirtualFree(mem_ptr, 0, MEM_RELEASE);
    mem_ptr = nullptr;
    return ReturnStatus::success;
}
#endif // __linux__

MemoryAllocator::MemoryAllocator() :
    m_imp(get_os_imp())
{
}

void* MemoryAllocator::allocate_aligned(size_t length, size_t alignment)
{
    byte_t* ptr = static_cast<byte_t*>(allocate(length + alignment));
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
