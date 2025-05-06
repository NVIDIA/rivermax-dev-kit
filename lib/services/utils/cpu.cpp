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

#include <iostream>
#include <thread>

#include <rivermax_api.h>
#include <rivermax_affinity.h>

#include "rt_threads.h"
#include "cpu.h"
#include "defs.h"

using namespace ral::lib::services;

ReturnStatus ral::lib::services::set_rivermax_thread_cpu_affinity(int cpu)
{
    if (cpu == CPU_NONE) {
        return ReturnStatus::success;
    }
    if (cpu < 0) {
        std::cerr << "Failed to mark CPU affinity to core " << cpu << std::endl;
        return ReturnStatus::failure;
    }

    constexpr size_t cores_per_mask = 8 * sizeof(uint64_t);
    std::vector<uint64_t> cpu_mask(cpu / cores_per_mask + 1, 0);
    rmx_mark_cpu_for_affinity(cpu_mask.data(), cpu);
    rmx_status status = rmx_set_cpu_affinity(cpu_mask.data(), size_t(cpu) + 1);
    if (status != RMX_OK) {
        std::cerr << "Failed to initialize Rivermax CPU affinity: " << status << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

void ral::lib::services::set_current_thread_affinity(const int cpu) 
{
    if (cpu != CPU_NONE) {
        rivermax::libs::set_affinity((size_t)cpu);
    } 
}
