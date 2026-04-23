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

#include <iostream>
#include <sched.h>
#include <stdexcept>
#include <string>

#include "rdk/services/cpu/affinity/linux/rivermax_os_affinity.h"
#include "rdk/services/cpu/affinity/rivermax_affinity.h"

namespace rdk
{
namespace services
{

LinuxAffinity::LinuxAffinity(const os_api &os_api)
    : m_os_api {os_api}
{
}

LinuxAffinity::editor::editor(const LinuxAffinity &affinity, std::thread::native_handle_type thread)
    : m_os_api {affinity.m_os_api}, m_thread {thread}
{
    m_cpu_set = m_os_api.cpu_alloc(cpu_mask_t::max_cpus);
    if (m_cpu_set == nullptr) {
        throw std::runtime_error("failed to allocate cpu_set for " + std::to_string(cpu_mask_t::max_cpus) + " cpus");
    }
    m_set_size = m_os_api.cpu_alloc_size(cpu_mask_t::max_cpus);
    m_os_api.cpu_zero_s(m_set_size, m_cpu_set);
}

void LinuxAffinity::editor::set(size_t processor)
{
    if (processor >= cpu_mask_t::max_cpus) {
        throw std::runtime_error("failed to apply illegal core number: " + std::to_string(processor)
            + " (must be < " + std::to_string(cpu_mask_t::max_cpus) + ")");
    }
    m_os_api.cpu_set(processor, m_cpu_set);
}

void LinuxAffinity::editor::apply()
{
    auto status = m_os_api.set_affinity_np(m_thread, m_set_size, m_cpu_set);
    if (status != 0) {
        throw std::runtime_error("failed setting thread affinity, errno: " + std::to_string(status));
    }
}

LinuxAffinity::editor::~editor()
{
    m_os_api.cpu_free(m_cpu_set);
}

size_t LinuxAffinity::count_cores() const
{
    return m_os_api.get_proc_count();
}

bool validate_thread_affinity_cpus(int /* internal_thread_affinity */, const std::vector<int>& /* cpus */)
{
    return true;
}

} // namespace services
} // namespace rdk
