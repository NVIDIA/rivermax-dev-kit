/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "rmax_receiver_base.h"

using namespace ral::lib::core;
using namespace ral::lib::services;
using namespace ral::apps;

RmaxReceiverBaseApp::RmaxReceiverBaseApp(const std::string& app_description, const std::string& app_examples) :
    RmaxBaseApp(app_description, app_examples)
{
}

ReturnStatus RmaxReceiverBaseApp::run()
{
    if (m_obj_init_status != ReturnStatus::obj_init_success) {
        return m_obj_init_status;
    }

    try {
        distribute_work_for_threads();
        configure_network_flows();
        initialize_receive_io_nodes();
        ReturnStatus rc = allocate_app_memory();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to allocate the memory required for the application" << std::endl;
            return rc;
        }
        rc = register_app_memory();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to register the memory on the NIC" << std::endl;
            return rc;
        }
        distribute_memory_for_receivers();
        run_receiver_threads();
    }
    catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

void RmaxReceiverBaseApp::distribute_work_for_threads()
{
    m_app_settings->num_of_threads = std::min<size_t>(m_app_settings->num_of_threads, m_app_settings->num_of_total_streams);
    m_streams_per_thread.reserve(m_app_settings->num_of_threads);
    for (int stream = 0; stream < m_app_settings->num_of_total_streams; stream++) {
        m_streams_per_thread[stream % m_app_settings->num_of_threads]++;
    }
}

bool RmaxReceiverBaseApp::allocate_and_align(size_t header_size, size_t payload_size, byte_t*& header, byte_t*& payload)
{
    header = payload = nullptr;
    if (header_size) {
        header = static_cast<byte_t*>(m_header_allocator->allocate_aligned(header_size,
                m_header_allocator->get_page_size()));
    }
    payload = static_cast<byte_t*>(m_payload_allocator->allocate_aligned(payload_size,
            m_payload_allocator->get_page_size()));
    return payload && (header_size == 0 || header);
}

ReturnStatus RmaxReceiverBaseApp::allocate_app_memory()
{
    ReturnStatus rc = get_total_streams_memory_size(m_header_mem_size, m_payload_mem_size);
    if (rc != ReturnStatus::success) {
        return rc;
    }

    bool alloc_successful = allocate_and_align(m_header_mem_size, m_payload_mem_size, m_header_buffer, m_payload_buffer);
    if (alloc_successful) {
        std::cout << "Allocated " << m_header_mem_size << " bytes for header"
            << " at address " << static_cast<void*>(m_header_buffer)
            << " and " <<  m_payload_mem_size << " bytes for payload"
            << " at address " << static_cast<void*>(m_payload_buffer) << std::endl;
    } else {
        std::cerr << "Failed to allocate memory" << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus RmaxReceiverBaseApp::cleanup_rivermax_resources()
{
    unregister_app_memory();
    return RmaxBaseApp::cleanup_rivermax_resources();
}
