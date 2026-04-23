/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <iostream>
#include <vector>

#include <rivermax_api.h>

#include "rdk/services/cpu/cpu_utils.h"
#include "rdk/services/cpu/affinity/rivermax_affinity.h"

bool rdk::services::set_rivermax_thread_cpu_affinity(int cpu)
{
    if (cpu == NO_CPU_AFFINITY) {
        return true;
    }
    if (cpu < 0) {
        std::cerr << "Invalid CPU core number: " << cpu << std::endl;
        return false;
    }

    Affinity::mask cpu_mask;
    rmx_mark_cpu_for_affinity(cpu_mask.bits, static_cast<size_t>(cpu));
    rmx_status status = rmx_set_cpu_affinity(cpu_mask.bits, static_cast<size_t>(cpu) + 1);
    if (status != RMX_OK) {
        std::cerr << "Failed to initialize Rivermax CPU affinity: " << status << std::endl;
        return false;
    }

    return true;
}

void rdk::services::set_current_thread_affinity(const int cpu)
{
    if (cpu == NO_CPU_AFFINITY) {
        return;
    }
    if (cpu < 0) {
        std::cerr << "Invalid CPU core number: " << cpu << std::endl;
        return;
    }
    auto status = set_affinity(static_cast<size_t>(cpu));
    if (!status) {
        std::cerr << "Failed to set CPU affinity to core " << cpu << std::endl;
    }
}

void rdk::services::set_current_thread_affinity(const std::vector<int> &cpus)
{
    Affinity::mask mask;
    bool has_affinity = false;
    for (auto cpu : cpus) {
        if (cpu == NO_CPU_AFFINITY) {
            continue;
        }
        if (cpu < 0 || static_cast<size_t>(cpu) >= Affinity::mask::max_cpus) {
            std::cerr << "Invalid CPU core number: " << cpu << " (skipped)" << std::endl;
            continue;
        }
        rmx_mark_cpu_for_affinity(mask.bits, static_cast<size_t>(cpu));
        has_affinity = true;
    }
    if (has_affinity) {
        auto status = set_affinity(mask);
        if (!status) {
            std::cerr << "Failed to set CPU affinity" << std::endl;
        }
    }
}
