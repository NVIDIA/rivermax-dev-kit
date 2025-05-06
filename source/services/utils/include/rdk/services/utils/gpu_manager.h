/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef RDK_SERVICES_UTILS_GPU_MANAGER_H_
#define RDK_SERVICES_UTILS_GPU_MANAGER_H_

#include <cstdint>

#include "rdk/services/error_handling/return_status.h"

#if defined(CUDA_ENABLED) && !defined(TEGRA_ENABLED)
#define CAN_USE_NVML 1
#else
#define CAN_USE_NVML 0
#endif

#if CAN_USE_NVML
#include <nvml.h>
#endif

namespace rivermax
{
namespace dev_kit
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
     * @brief: Returns graphics clock's maximum frequency.
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
     * @brief: Returns memory clock's maximum frequency.
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
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_UTILS_GPU_MANAGER_H_ */
