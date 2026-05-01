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

#include <string>
#include <cstring>

#include "rt_threads.h"

#include "rdk/apps/base_app.h"

using namespace rdk::apps;
using namespace rdk::services;

namespace {
static const std::map<AllocatorTypeUI, AllocatorType> UI_ALLOCATOR_TYPE_MAP{
    { AllocatorTypeUI::Auto,            AllocatorType::HugePageDefault },
    { AllocatorTypeUI::HugePageDefault, AllocatorType::HugePageDefault },
    { AllocatorTypeUI::Malloc,          AllocatorType::Malloc },
    { AllocatorTypeUI::HugePage2MB,     AllocatorType::HugePage2MB },
    { AllocatorTypeUI::HugePage512MB,   AllocatorType::HugePage512MB },
    { AllocatorTypeUI::HugePage1GB,     AllocatorType::HugePage1GB },
#ifdef CUDA_ENABLED
    { AllocatorTypeUI::GPUHostPinned,      AllocatorType::GPUHostPinned }
#endif
};
}
BaseApp::BaseApp() :
    m_obj_init_status(ReturnStatus::obj_init_failure),
    m_app_settings(nullptr),
    m_rivermax_dev_kit(RivermaxDevKitFacade::get_instance()),
    m_device_interfaces{},
    m_stats_reader(nullptr),
    m_memory_utils(nullptr)
{
    m_signal_handler = m_rivermax_dev_kit.get_signal_handler(true);
    m_gpu_manager = m_rivermax_dev_kit.get_gpu_manager();
}

BaseApp::~BaseApp()
{
    if (m_obj_init_status != ReturnStatus::obj_init_success) {
        return;
    }

    for (auto& thread : m_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    cleanup_rivermax_resources();
}

ReturnStatus BaseApp::initialize_memory_allocators()
{
    const auto alloc_type_iter = UI_ALLOCATOR_TYPE_MAP.find(m_app_settings->allocator_type);
    if (alloc_type_iter == UI_ALLOCATOR_TYPE_MAP.end()) {
        std::cerr << "Unknown UI allocator type " << static_cast<int>(m_app_settings->allocator_type) << std::endl;
        return ReturnStatus::failure;
    }
    AllocatorType allocator_type = alloc_type_iter->second;
    AllocatorType payload_allocator_type;
    if (m_app_settings->gpu_id != INVALID_GPU_ID) {
        payload_allocator_type = AllocatorType::GPU;
    } else {
        payload_allocator_type = allocator_type;
    }
    m_header_allocator = m_rivermax_dev_kit.get_memory_allocator(allocator_type, m_app_settings);
    if (m_header_allocator == nullptr) {
        std::cerr << "Failed to create header memory allocator" << std::endl;
        return ReturnStatus::failure;
    }
    m_payload_allocator = m_rivermax_dev_kit.get_memory_allocator(payload_allocator_type, m_app_settings);
    if (m_payload_allocator == nullptr) {
        std::cerr << "Failed to create payload memory allocator" << std::endl;
        return ReturnStatus::failure;
    }
    m_auxiliary_allocator = m_rivermax_dev_kit.get_memory_allocator(allocator_type, m_app_settings);
    if (m_auxiliary_allocator == nullptr) {
        std::cerr << "Failed to create auxiliary memory allocator" << std::endl;
        return ReturnStatus::failure;
    }
    m_memory_utils = std::make_unique<IONodeMemoryUtils>(*m_header_allocator, *m_payload_allocator);
    return ReturnStatus::success;
}

ReturnStatus BaseApp::initialize_gpu_manager()
{
    if (m_app_settings->gpu_id == INVALID_GPU_ID) {
        return ReturnStatus::success;
    }
    const ReturnStatus status = m_gpu_manager->initialize(m_app_settings->gpu_id);
    if (status != ReturnStatus::success) {
        std::cerr << "Failed to initialize GPU manager" << std::endl;
        return status;
    }
    if (m_app_settings->lock_gpu_clocks) {
        ReturnStatus clock_status = m_gpu_manager->lock_clocks_max_frequency();
        if (clock_status != ReturnStatus::success) {
            std::cerr << "Warning - Failed to lock GPU clocks to their maximum frequency. "
                         "Performance might be affected."
                      << std::endl;
        }
    }
    return status;
}

ReturnStatus BaseApp::initialize()
{
    ReturnStatus rc = initialize_app_settings();
    if (rc == ReturnStatus::success_cli_help) {
        return rc;
    } else if (rc !=  ReturnStatus::success || m_app_settings == nullptr) {
        std::cerr << "Failed to initialize application settings" << std::endl;
        return rc;
    }

    rc = post_load_settings();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to post load settings" << std::endl;
        return rc;
    }

    rc = initialize_gpu_manager();
    if (rc == ReturnStatus::failure) {
        return rc;
    }

    rc = initialize_memory_allocators();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to initialize memory allocators" << std::endl;
        m_obj_init_status = ReturnStatus::memory_allocation_failure;
        return m_obj_init_status;
    }

    rc = initialize_rivermax_resources();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to initialize Rivermax resources" << std::endl;
        return rc;
    }

    rc = initialize_connection_parameters();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to initialize application connection parameters" << std::endl;
        return rc;
    }

    rc = set_rivermax_clock();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to set Rivermax clock" << std::endl;
        return rc;
    }

    return ReturnStatus::success;
}

ReturnStatus BaseApp::initialize_rivermax_resources()
{
    rt_set_realtime_class();
    ReturnStatus rc = m_rivermax_dev_kit.initialize_rivermax(
        m_rivermax_lib_handle, m_app_settings->internal_thread_core);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to initialize Rivermax resources" << std::endl;
        return rc;
    }
    return ReturnStatus::success;
}

ReturnStatus BaseApp::set_rivermax_clock()
{
    return ReturnStatus::success;
}

ReturnStatus BaseApp::initialize_connection_parameters()
{
    m_local_addresses.clear();
    m_device_interfaces.clear();

    std::vector<std::string> local_ips = m_app_settings->local_ips;
    if (local_ips.empty()) {
        local_ips.push_back(m_app_settings->local_ip);
    }

    for (const auto& local_ip : local_ips) {
        sockaddr_in local_address;
        memset(&local_address, 0, sizeof(sockaddr_in));
        local_address.sin_family = AF_INET;
        int rc = inet_pton(AF_INET, local_ip.c_str(), &local_address.sin_addr);
        if (rc != 1) {
            std::cerr << "Failed to parse local network address: " << local_ip << std::endl;
            return ReturnStatus::failure;
        }
        m_local_addresses.push_back(local_address);
        rmx_device_iface device_interface;
        rmx_status status = rmx_retrieve_device_iface_ipv4(&device_interface, &local_address.sin_addr);
        if (status != RMX_OK) {
            std::cerr << "Failed to get device: " << local_ip << " with status: " << status << std::endl;
            return ReturnStatus::failure;
        }
        m_device_interfaces.push_back(device_interface);
    }

    return ReturnStatus::success;
}

ReturnStatus BaseApp::configure_memory_layout()
{
    ReturnStatus rc = initialize_memory_strategy();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to initialize memory strategy" << std::endl;
        return rc;
    }
    if (!m_memory_strategy) {
        std::cerr << "Memory strategy is not initialized" << std::endl;
        return ReturnStatus::failure;
    }
    rc = m_memory_strategy->determine_memory_layout();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to determine memory layout" << std::endl;
        return rc;
    }
    rc = m_memory_strategy->allocate_memory();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to allocate the memory required for the application" << std::endl;
        return rc;
    }
    rc = m_memory_strategy->register_memory();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to register the memory on the NIC" << std::endl;
        return rc;
    }
    rc = m_memory_strategy->apply_memory_layout();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to apply memory layout" << std::endl;
        return rc;
    }
    return ReturnStatus::success;
}

ReturnStatus BaseApp::run_stats_reader()
{
    if (!is_run_stats_reader()) {
        return ReturnStatus::success;
    }

    m_stats_reader.reset(new StatisticsReader());

    ReturnStatus rc = m_stats_reader->init_stats_reader();
    if (rc == ReturnStatus::failure) {
        m_stats_reader.reset();
        std::cerr << "Failed to run StatisticsReader\n"
                     "Ensure that the producer exists and that RIVERMAX_ENABLE_STATISTICS is set" << std::endl;
        return ReturnStatus::failure;
    }

    if (m_app_settings->statistics_reader_core != INVALID_CORE_NUMBER) {
        m_stats_reader->set_cpu_core_affinity(m_app_settings->statistics_reader_core);
    }
    if (m_app_settings->session_id_stats != UINT_MAX) {
        std::cout << "Set presen session id: " << m_app_settings->session_id_stats << std::endl;
        m_stats_reader->set_session_id(m_app_settings->session_id_stats);
    }

    m_threads.push_back(std::thread(std::ref(*m_stats_reader)));
    return ReturnStatus::success;
}

ReturnStatus BaseApp::cleanup_rivermax_resources()
{
    if (m_memory_strategy) {
        m_memory_strategy->unregister_memory();
    }
    m_rivermax_lib_handle.reset();

    return ReturnStatus::success;
}

ReturnStatus BaseApp::find_internal_stream_index(
    size_t external_stream_index, size_t& thread_index, size_t& internal_stream_index)
{
    if (external_stream_index >= m_app_settings->num_of_total_streams) {
        std::cerr << "Error: Invalid stream index " << external_stream_index << std::endl;
        return ReturnStatus::failure;
    }

    size_t first_stream_index = 0;
    for (thread_index = 0; thread_index < m_app_settings->num_of_threads; thread_index++) {
        const size_t streams_in_thread = m_streams_per_thread[thread_index];
        const size_t last_stream_index_in_thread = first_stream_index + streams_in_thread;
        if (external_stream_index < last_stream_index_in_thread) {
            internal_stream_index = external_stream_index - first_stream_index;
            return ReturnStatus::success;
        }
        first_stream_index += streams_in_thread;
    }

    std::cerr << "Error: Could not find internal indexes for stream index " << external_stream_index << std::endl;
    return ReturnStatus::failure;
}

ReturnStatus BaseApp::get_app_settings(const AppSettings*& settings) const {
    if (m_obj_init_status != ReturnStatus::obj_init_success || !m_app_settings) {
        std::cerr << "Error: Application settings are not initialized" << std::endl;
        settings = nullptr;
        return ReturnStatus::failure;
    }
    settings = m_app_settings.get();
    return ReturnStatus::success;
}
