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

#include <cstdint>
#include <iostream>
#include <vector>

#include <rivermax_api.h>

#include "rdk/services/cpu/affinity/affinity_utils.h"
#include "rdk/services/cpu/affinity/rivermax_affinity.h"

bool rivermax::dev_kit::services::set_rivermax_thread_cpu_affinity(int cpu)
{
    if (cpu == INVALID_CORE_NUMBER) {
        return true;
    }
    if (cpu < 0) {
        std::cerr << "Invalid CPU core number: " << cpu << std::endl;
        return false;
    }

    constexpr size_t cores_per_mask = 8 * sizeof(uint64_t);
    std::vector<uint64_t> cpu_mask(cpu / cores_per_mask + 1, 0);
    rmx_mark_cpu_for_affinity(cpu_mask.data(), cpu);
    rmx_status status = rmx_set_cpu_affinity(cpu_mask.data(), size_t(cpu) + 1);
    if (status != RMX_OK) {
        std::cerr << "Failed to initialize Rivermax CPU affinity: " << status << std::endl;
        return false;
    }

    return true;
}

void rivermax::dev_kit::services::set_current_thread_affinity(const int cpu)
{
    if (cpu == INVALID_CORE_NUMBER) {
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
