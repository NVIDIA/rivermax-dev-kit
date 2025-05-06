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

#include <cassert>

#include "rdk/apps/rmax_receiver_base.h"
#include "rdk/apps/rmax_base_memory_strategy.h"

using namespace rivermax::dev_kit::apps;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

RmaxReceiverBaseApp::RmaxReceiverBaseApp() :
    RmaxBaseApp()
{
}

ReturnStatus RmaxReceiverBaseApp::initialize()
{
    ReturnStatus rc  = RmaxBaseApp::initialize();

    if (rc != ReturnStatus::obj_init_success) {
        return m_obj_init_status;
    }

    try {
        distribute_work_for_threads();
        configure_network_flows();
        initialize_receive_io_nodes();
        rc = configure_memory_layout();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to configure memory layout" << std::endl;
            return rc;
        }
    }
    catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return ReturnStatus::failure;
    }

    m_obj_init_status = ReturnStatus::obj_init_success;
    return m_obj_init_status;
}

ReturnStatus RmaxReceiverBaseApp::run()
{
    if (m_obj_init_status != ReturnStatus::obj_init_success) {
        return m_obj_init_status;
    }

    try {
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

ReturnStatus RmaxReceiverBaseApp::initialize_memory_strategy()
{
    auto base_memory_strategy = std::make_unique<RmaxBaseMemoryStrategy>(
        *m_header_allocator, *m_payload_allocator,
        *m_memory_utils,
        m_device_interfaces,
        m_num_paths_per_stream,
        m_app_settings->app_memory_alloc,
        m_app_settings->register_memory);

    for (const auto& receiver : m_receivers) {
        base_memory_strategy->add_memory_subcomponent(receiver);
    }

    m_memory_strategy.reset(base_memory_strategy.release());

    return ReturnStatus::success;
}

ReturnStatus RmaxReceiverBaseApp::set_receive_data_consumer(
    size_t stream_index, std::unique_ptr<IReceiveDataConsumer> data_consumer)
{
    size_t receiver_thread_index = 0;
    size_t receiver_stream_index = 0;

    auto rc = find_internal_stream_index(stream_index, receiver_thread_index, receiver_stream_index);
    if (rc != ReturnStatus::success) {
        std::cerr << "Error setting data consumer, invalid stream index " << stream_index << std::endl;
        return rc;
    }

    rc = m_receivers[receiver_thread_index]->set_receive_data_consumer(
        receiver_stream_index, std::move(data_consumer));

    if (rc != ReturnStatus::success) {
        std::cerr << "Error setting data consumer for stream "
                  << receiver_stream_index << " on receiver " << receiver_thread_index << std::endl;
    }

    return rc;
}
