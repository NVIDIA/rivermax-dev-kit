/*
 * Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
