/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "rivermax_os_affinity.h"
#include <stdexcept>
#include <string>

namespace rivermax
{
namespace libs
{

LinuxAffinity::LinuxAffinity(const os_api &os_api)
    : m_os_api {os_api} 
{
}

LinuxAffinity::editor::editor(const LinuxAffinity &affinity, std::thread::native_handle_type thread) 
    : m_os_api {affinity.m_os_api}, m_thread {thread}
{
    m_cpu_set = m_os_api.cpu_alloc(RMAX_CPU_SETSIZE);
    if (m_cpu_set == nullptr) {
        throw std::runtime_error("failed to allocate cpu_set for " + std::to_string(RMAX_CPU_SETSIZE) + " cpus");
    }
    m_set_size = m_os_api.cpu_alloc_size(RMAX_CPU_SETSIZE);
    m_os_api.cpu_zero_s(m_set_size, m_cpu_set);
}

void LinuxAffinity::editor::set(size_t processor) 
{
    if (processor >= RMAX_CPU_SETSIZE) {
        throw std::runtime_error("failed to apply illegal core number: " + std::to_string(processor) );
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

} // namespace libs
} // namespace rivermax
