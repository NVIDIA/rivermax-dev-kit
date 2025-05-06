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

#ifndef RMAX_APPS_LIB_LIB_SERVICES_UTILS_GPU_MANAGER_H_
#define RMAX_APPS_LIB_LIB_SERVICES_UTILS_GPU_MANAGER_H_

#include "services/error_handling/return_status.h"

#include <cstdint>

#if defined(CUDA_ENABLED) && !defined(TEGRA_ENABLED)
#define CAN_USE_NVML 1
#else
#define CAN_USE_NVML 0
#endif

#if CAN_USE_NVML
#include <nvml.h>
#endif

namespace ral
{
namespace lib
{
namespace services
{

constexpr int GPU_ID_INVALID = -1;
constexpr const char* CUDA_DEVICE_ORDER = "CUDA_DEVICE_ORDER";
constexpr const char* CUDA_PCI_BUS_ID_DEVICE_ORDER = "PCI_BUS_ID";

/**
 * @brief: Holds information about a memory section.
 */
struct gpu_memory_region_info
{
    uint64_t free = 0;
    uint64_t total = 0;
    uint64_t used = 0;
};

/**
 * @brief: GPU Manager class.
 *
 * This class includes utilities to manage GPU settings
 * or monitor GPU configuration. It is mainly used to
 * wrap NVML library methods.
 */
class GPUManager
{
public:
    /**
     * @brief: GPUManager default constructor.
     */
    GPUManager() = default;
    ~GPUManager();
    /**
     * @brief: Initializes GPU management internals.
     *
     * @param [in] gpu_id: ID of GPU to configure.
     *
     * @return: Status of the operation.
     */
    ReturnStatus initialize(int gpu_id);
    /**
     * @brief: Locks GPU graphics and memory clocks to their maximum frequency.
     *
     * @return: Status of the operation.
     */
    ReturnStatus lock_clocks_max_frequency();
    /**
     * @brief: Unlocks GPU graphics and memory clocks.
     *
     * @return: Status of the operation.
     */
    ReturnStatus unlock_clocks();
    /**
     * @brief: Locks graphics clock frequency.
     *
     * @param [out] frequency: Graphics clock frequency.
     *
     * @return: Status of the operation.
     */
    ReturnStatus lock_graphics_clock(uint32_t frequency);
    /**
     * @brief: Unlocks graphics clock frequency.
     *
     * @return: Status of the operation.
     */
    ReturnStatus unlock_graphics_clock();
    /**
     * @brief: Gets graphics clock's maximum frequency.
     *
     * @param [out] frequency: Graphics clock maximum frequency.
     *
     * @return: Status of the operation.
     */
    ReturnStatus get_graphics_clock_max_frequency(uint32_t& frequency) const;
    /**
     * @brief: Locks memory clock frequency.
     *
     * @param [out] frequency: Memory clock frequency.
     *
     * @return: Status of the operation.
     */
    ReturnStatus lock_memory_clock(uint32_t frequency);
    /**
     * @brief: Unlocks memory clock frequency.
     *
     * @param [out] frequency: Memory clock max frequency.
     *
     * @return: Status of the operation.
     */
    ReturnStatus unlock_memory_clock();
    /**
     * @brief: Gets memory clock's maximum frequency.
     *
     * @param [out] frequency: Memory clock maximum frequency.
     *
     * @return: Status of the operation.
     */
    ReturnStatus get_memory_clock_max_frequency(uint32_t& frequency) const;
    /**
     * @brief: Queries GPU BAR1 memory information.
     *
     * @param [out] info: BAR1 memory information.
     *
     * @return: Status of the operation.
     */
    ReturnStatus get_bar1_memory_info(gpu_memory_region_info& info) const;
private:
    /**
     * @brief: Initializes NVML library and validates given GPU ID.
     *
     * @return: Returns status of operation.
     */
    ReturnStatus initialize_nvml();
    /**
     * @brief: Shuts down NVML library.
     */
    void shutdown_nvml();
private:
    bool m_is_graphics_clock_locked = false;
    bool m_is_memory_clock_locked = false;
    bool m_is_initialized = false;
    int m_gpu_id = GPU_ID_INVALID;

#if CAN_USE_NVML
    nvmlDevice_t m_device_handle = nullptr;
#endif
};

} // namespace services
} // namespace lib
} // namespace ral

#endif /* RMAX_APPS_LIB_LIB_SERVICES_UTILS_GPU_MANAGER_H_ */
