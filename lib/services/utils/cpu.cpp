/*
 * Copyright © 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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

#include "rt_threads.h"
#include "rivermax_api.h"
#include "cpu.h"

using namespace ral::lib::services;

ReturnStatus ral::lib::services::set_rivermax_thread_cpu_affinity(const std::vector<int>& cpu_affinity)
{
    auto n_cpus = static_cast<int>(std::thread::hardware_concurrency());
    std::vector<uint64_t> cpu_mask((n_cpus + sizeof(uint64_t) - 1) / sizeof(uint64_t), 0);
    bool affinity_restricted = false;

    for (auto cpu : cpu_affinity) {
        if (cpu == CPU_NONE) {
            continue;
        }
        if (cpu >= n_cpus || cpu < 0) {
            std::cerr << "Failed to mark CPU affinity to core " << cpu << std::endl;
            return ReturnStatus::failure;
        }
        rmx_mark_cpu_for_affinity(cpu_mask.data(), cpu);
        affinity_restricted = true;
    }

    if (!affinity_restricted) {
        return ReturnStatus::success;
    }

    rmx_status status = rmx_set_cpu_affinity(cpu_mask.data(), n_cpus);
    if (status != RMX_OK) {
        std::cerr << "Failed to initialize Rivermax CPU affinity: " << status << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus ral::lib::services::set_app_thread_cpu_affinity(const std::vector<int>& cpu_affinity)
{
    set_cpu_affinity(cpu_affinity);

    return ReturnStatus::success;
}
