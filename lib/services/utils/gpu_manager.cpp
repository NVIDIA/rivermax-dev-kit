/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "gpu_manager.h"

#include "defs.h"
#include "rt_threads.h"

#include <iostream>

using namespace ral::lib::services;

GPUManager::~GPUManager()
{
    unlock_clocks();
    shutdown_nvml();
}

ReturnStatus GPUManager::initialize(int gpu_id)
{
    if (gpu_id == GPU_ID_INVALID) {
        return ReturnStatus::failure;
    }
    m_gpu_id = gpu_id;

    /**
     * nvidia-smi is the user's primary tool for identifying the ID of the GPU which is to be used
     * by the Rivermax application. The ID returned by nvidia-smi is obtained by its enumerating the
     * GPU devices according to their PCI order. However, by default the CUDA driver and runtime
     * APIs enumerate GPUs according to their speed (and not by PCI order). To align the ID obtained
     * from nvidia-smi and those used by the CUDA driver and runtime APIs we modify their GPU ID
     * enumeration policy by setting the CUDA_DEVICE_ORDER environment variable with value
     * PCI_BUS_ID.
     */
    const int ret = set_enviroment_variable(CUDA_DEVICE_ORDER, CUDA_PCI_BUS_ID_DEVICE_ORDER);
    if (ret != 0) {
        std::cerr << "Failed to set env variable " << CUDA_DEVICE_ORDER << "="
                  << CUDA_PCI_BUS_ID_DEVICE_ORDER << std::endl;
        return ReturnStatus::failure;
    }
    std::cout << "Set env variable " << CUDA_DEVICE_ORDER << "=" << CUDA_PCI_BUS_ID_DEVICE_ORDER
              << std::endl;

    return initialize_nvml();
}

#if CAN_USE_NVML

ReturnStatus GPUManager::initialize_nvml()
{
    nvmlReturn_t nv_status = nvmlInit();
    if (nv_status != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVIDIA Management Library (NVML) with error: "
                  << nv_status << std::endl;
        return ReturnStatus::failure;
    }
    m_is_initialized = true;

    nvmlDevice_t nv_device;
    nv_status = nvmlDeviceGetHandleByIndex(m_gpu_id, &nv_device);
    if (nv_status != NVML_SUCCESS) {
        std::cerr << "Failed to get GPU device " << m_gpu_id << " with error: " << nv_status
                  << std::endl;
        return ReturnStatus::failure;
    }
    m_device_handle = nv_device;

    return ReturnStatus::success;
}

void GPUManager::shutdown_nvml()
{
    if (m_is_initialized) {
        const nvmlReturn_t nv_status = nvmlShutdown();
        if (nv_status != NVML_SUCCESS) {
            std::cerr << "Failed to shutdown NVIDIA Management Library (NVML) with error: "
                      << nv_status << std::endl;
        }
    }
}

ReturnStatus GPUManager::lock_clocks_max_frequency()
{
    uint32_t max_graphics_clock_freq = 0;
    uint32_t max_memory_clock_freq = 0;
    ReturnStatus status = get_graphics_clock_max_frequency(max_graphics_clock_freq);
    if (status != ReturnStatus::success) {
        return status;
    }
    status = get_memory_clock_max_frequency(max_memory_clock_freq);
    if (status != ReturnStatus::success) {
        return status;
    }
    status = lock_graphics_clock(max_graphics_clock_freq);
    if (status != ReturnStatus::success) {
        return status;
    }
    status = lock_memory_clock(max_memory_clock_freq);
    if (status != ReturnStatus::success) {
        unlock_graphics_clock();
    }

    return status;
}

ReturnStatus GPUManager::unlock_clocks()
{
    const ReturnStatus gpu_status = unlock_graphics_clock();
    const ReturnStatus mem_status = unlock_memory_clock();
    return gpu_status != ReturnStatus::success ? gpu_status : mem_status;
}

ReturnStatus GPUManager::lock_graphics_clock(uint32_t frequency)
{
    const nvmlReturn_t nv_status =
        nvmlDeviceSetGpuLockedClocks(m_device_handle, frequency, frequency);
    if (nv_status != NVML_SUCCESS) {
        if (nv_status == NVML_ERROR_NOT_SUPPORTED) {
            std::cout << "Warning! Setting locked GPU clock is not supported" << std::endl;
        } else {
            std::cerr << "Failed to set GPU clock on max frequency with error: " << nv_status
                      << std::endl;
        }
        return ReturnStatus::failure;
    }

    m_is_graphics_clock_locked = true;
    return ReturnStatus::success;
}

ReturnStatus GPUManager::unlock_graphics_clock()
{
    if(!m_is_graphics_clock_locked) {
        return ReturnStatus::success;
    }

    ReturnStatus status = ReturnStatus::success;
    const nvmlReturn_t nv_status = nvmlDeviceResetGpuLockedClocks(m_device_handle);
    if (nv_status != NVML_SUCCESS) {
        if (nv_status == NVML_ERROR_NOT_SUPPORTED) {
            std::cout << "Warning! Resetting locked GPU clock is not supported" << std::endl;
            status = ReturnStatus::failure;
        } else {
            std::cerr << "Failed to reset GPU clock to default frequency with error: " << nv_status
                      << std::endl;
            status = ReturnStatus::failure;
        }
    }

    m_is_graphics_clock_locked = false;
    return status;
}

ReturnStatus GPUManager::get_graphics_clock_max_frequency(uint32_t& frequency) const
{
    uint32_t max_frequency = 0;
    const nvmlReturn_t nv_status =
        nvmlDeviceGetMaxClockInfo(m_device_handle, NVML_CLOCK_GRAPHICS, &max_frequency);
    if (nv_status != NVML_SUCCESS) {
        std::cerr << "Failed to get max graphics clock frequency with error: " << nv_status
                  << std::endl;
        return ReturnStatus::failure;
    }

    frequency = max_frequency;
    return ReturnStatus::success;
}

ReturnStatus GPUManager::lock_memory_clock(uint32_t frequency)
{
    const nvmlReturn_t nv_status =
        nvmlDeviceSetMemoryLockedClocks(m_device_handle, frequency, frequency);
    if (nv_status != NVML_SUCCESS) {
        if (nv_status == NVML_ERROR_NOT_SUPPORTED) {
            std::cout << "Warning! Setting locked GPU clock is not supported" << std::endl;
        } else {
            std::cerr << "Failed to set GPU clock on max frequency with error: " << nv_status
                      << std::endl;
        }
        return ReturnStatus::failure;
    }

    m_is_memory_clock_locked = true;
    return ReturnStatus::success;
}

ReturnStatus GPUManager::unlock_memory_clock()
{
    if(!m_is_memory_clock_locked) {
        return ReturnStatus::success;
    }
    
    ReturnStatus status = ReturnStatus::success;
    const nvmlReturn_t nv_status = nvmlDeviceResetMemoryLockedClocks(m_device_handle);
    if (nv_status != NVML_SUCCESS) {
        if (nv_status == NVML_ERROR_NOT_SUPPORTED) {
            std::cout << "Warning! Resetting locked GPU memory clock is not supported" << std::endl;
            status = ReturnStatus::failure;
        } else {
            std::cerr << "Failed to reset GPU memory clock to default frequency with error: "
                      << nv_status << std::endl;
            status = ReturnStatus::failure;
        }
    }

    m_is_memory_clock_locked = false;
    return status;
}

ReturnStatus GPUManager::get_memory_clock_max_frequency(uint32_t& frequency) const
{
    uint32_t max_frequency = 0;
    const nvmlReturn_t nv_status =
        nvmlDeviceGetMaxClockInfo(m_device_handle, NVML_CLOCK_MEM, &max_frequency);
    if (nv_status != NVML_SUCCESS) {
        std::cerr << "Failed to get max memory clock frequency with error: " << nv_status
                  << std::endl;
        return ReturnStatus::failure;
    }

    frequency = max_frequency;
    return ReturnStatus::success;
}

ReturnStatus GPUManager::get_bar1_memory_info(gpu_memory_region_info& info) const
{
    nvmlBAR1Memory_t nv_bar_memory;
    const nvmlReturn_t nv_status = nvmlDeviceGetBAR1MemoryInfo(m_device_handle, &nv_bar_memory);
    if (nv_status != NVML_SUCCESS) {
        std::cerr << "Failed to get GPU BAR1 memory information with error: " << nv_status
                  << std::endl;
        return ReturnStatus::failure;
    }

    info.free = nv_bar_memory.bar1Free;
    info.total = nv_bar_memory.bar1Total;
    info.used = nv_bar_memory.bar1Used;

    return ReturnStatus::success;
}

#else //CAN_USE_NVML

ReturnStatus GPUManager::initialize_nvml()
{
    return ReturnStatus::success;
}

void GPUManager::shutdown_nvml()
{
}

ReturnStatus GPUManager::lock_clocks_max_frequency()
{
    return ReturnStatus::success;
}

ReturnStatus GPUManager::unlock_clocks()
{
    return ReturnStatus::success;
}

ReturnStatus GPUManager::lock_graphics_clock(uint32_t frequency)
{
    return ReturnStatus::success;
}

ReturnStatus GPUManager::unlock_graphics_clock()
{
    return ReturnStatus::success;
}

ReturnStatus GPUManager::get_graphics_clock_max_frequency(uint32_t& frequency) const
{
    return ReturnStatus::success;
}

ReturnStatus GPUManager::lock_memory_clock(uint32_t frequency)
{
    return ReturnStatus::success;
}

ReturnStatus GPUManager::unlock_memory_clock()
{
    return ReturnStatus::success;
}

ReturnStatus GPUManager::get_memory_clock_max_frequency(uint32_t& frequency) const
{
    return ReturnStatus::success;
}

ReturnStatus GPUManager::get_bar1_memory_info(gpu_memory_region_info& info) const
{
    return ReturnStatus::success;
}

#endif //CAN_USE_NVML
