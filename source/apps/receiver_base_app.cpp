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

#include <cassert>

#include "rdk/apps/receiver_base_app.h"
#include "rdk/apps/base_memory_strategy.h"
#include "rdk/io_node/receivers/rtp_receiver_io_node.h"
#include "rdk/io_node/receivers/ipo_receiver_io_node.h"

using namespace rdk::apps;
using namespace rdk::services;
using namespace rdk::core;

template<typename IONodeSettingsType>
ReceiverBaseApp<IONodeSettingsType>::ReceiverBaseApp() :
    BaseApp()
{
}

template<typename IONodeSettingsType>
ReturnStatus ReceiverBaseApp<IONodeSettingsType>::initialize()
{
    ReturnStatus rc  = BaseApp::initialize();
    if (rc != ReturnStatus::success) {
        return rc;
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
    return ReturnStatus::success;
}

template<typename IONodeSettingsType>
ReturnStatus ReceiverBaseApp<IONodeSettingsType>::do_work()
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

template<typename IONodeSettingsType>
void ReceiverBaseApp<IONodeSettingsType>::stop_io_nodes() noexcept
{
    BaseApp::stop_io_nodes(m_receivers);
}

template<typename IONodeSettingsType>
size_t ReceiverBaseApp<IONodeSettingsType>::num_io_nodes() const
{
    return m_receivers.size();
}

template<typename IONodeSettingsType>
ReturnStatus ReceiverBaseApp<IONodeSettingsType>::stop_io_node(size_t id)
{
    return BaseApp::stop_io_node(m_receivers, id);
}

template<typename IONodeSettingsType>
ReturnStatus ReceiverBaseApp<IONodeSettingsType>::await_io_node(size_t id, std::chrono::nanoseconds timeout) const
{
    return BaseApp::await_io_node(m_receivers, id, timeout);
}

template<typename IONodeSettingsType>
void ReceiverBaseApp<IONodeSettingsType>::distribute_work_for_threads()
{
    m_app_settings->num_of_threads = std::min<size_t>(m_app_settings->num_of_threads, m_app_settings->num_of_total_streams);
    m_streams_per_thread.reserve(m_app_settings->num_of_threads);
    for (int stream = 0; stream < m_app_settings->num_of_total_streams; stream++) {
        m_streams_per_thread[stream % m_app_settings->num_of_threads]++;
    }
}

template<typename IONodeSettingsType>
ReturnStatus ReceiverBaseApp<IONodeSettingsType>::initialize_memory_strategy()
{
    auto base_memory_strategy = std::make_unique<BaseMemoryStrategy>(
        *m_header_allocator, *m_payload_allocator, *m_auxiliary_allocator,
        *m_memory_utils,
        m_device_interfaces,
        m_num_paths_per_stream,
        m_app_settings->app_memory_alloc,
        m_app_settings->register_memory,
        m_app_settings->allocate_extra_stream_buffers);

    for (const auto& receiver : m_receivers) {
        base_memory_strategy->add_memory_subcomponent(receiver);
    }

    m_memory_strategy = std::move(base_memory_strategy);

    return ReturnStatus::success;
}

template<typename IONodeSettingsType>
ReturnStatus ReceiverBaseApp<IONodeSettingsType>::set_receive_data_consumer(
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

template class ReceiverBaseApp<RTPReceiverIONodeSettings>;
template class ReceiverBaseApp<IPOReceiverIONodeSettings>;
