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
#include <cstddef>
#include <cstring>
#include <iostream>
#include <ostream>
#include <thread>
#include <vector>

#include <rivermax_api.h>
#include "rt_threads.h"

#include "rdk/io_node/receivers/rtp_receiver_io_node.h"
#include "rdk/services/cpu/cpu.h"
#include "rdk/services/error_handling/error_handling.h"

using namespace rdk::io_node;
using namespace rdk::services;
using namespace rdk::core;

ReceiverIONodeBase::ReceiverIONodeBase(
        const AppSettings& app_settings,
        size_t index, int cpu_core_affinity,
        IONodeMemoryUtils& memory_utils) :
    m_app_settings(app_settings),
    m_index(index),
    m_print_parameters(app_settings.print_parameters),
    m_cpu_core_affinity(cpu_core_affinity),
    m_sleep_between_operations(std::chrono::microseconds(app_settings.sleep_between_operations_us)),
    m_memory_utils(memory_utils)
{
}

std::ostream& ReceiverIONodeBase::print(std::ostream& out) const
{
    out << "+#############################################\n"
        << "| Receiver index: " << m_index << "\n"
        << "| Thread ID: 0x" << std::hex << std::this_thread::get_id() << std::dec << "\n"
        << "| CPU core affinity: " << m_cpu_core_affinity << "\n"
        << "| Number of streams in this thread: " << m_streams.size() << "\n"
        << "+---------------------------------------------\n";
    for (const auto& stream : m_streams) {
        stream->print(out);
        out << "+---------------------------------------------\n";
    }
    return out;
}

void ReceiverIONodeBase::print_parameters() const
{
    if (!m_print_parameters) {
        return;
    }

    std::stringstream receiver_parameters;
    print(receiver_parameters);
    std::cout << receiver_parameters.str() << std::endl;
}

void ReceiverIONodeBase::set_statistics_report_interval(uint32_t print_interval_ms)
{
    m_print_interval_ms = print_interval_ms;
}

ReturnStatus ReceiverIONodeBase::create_streams()
{
    ReturnStatus rc;
    for (size_t i = 0; i < m_streams.size(); ++i) {
        const auto& stream = m_streams[i];
        rc = stream->create_stream();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to create stream (" << stream->get_stream_name() << ")" << std::endl;
            return rc;
        }
        auto chunk = create_chunk(i);
        m_stream_chunks.emplace_back(std::move(chunk));
    }

    return ReturnStatus::success;
}

ReturnStatus ReceiverIONodeBase::destroy_streams()
{
    ReturnStatus rc;

    for (auto& stream : m_streams) {
        rc = stream->destroy_stream();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to destroy stream (" << stream->get_stream_name() << ")" << std::endl;
            return rc;
        }
    }

    return ReturnStatus::success;
}

ReturnStatus ReceiverIONodeBase::initialize_memory_layout()
{
    size_t header_memory_size = 0;
    size_t payload_memory_size = 0;
    size_t auxiliary_memory_size = 0;
    m_header_total_memory_size = 0;
    m_payload_total_memory_size = 0;
    m_auxiliary_total_memory_size = 0;

    m_buffer_sizes.clear();
    m_buffer_sizes.reserve(m_streams.size());

    for (const auto& stream : m_streams) {
        StreamMemoryLayoutRequest stream_memory_layout;
        ReturnStatus rc = stream->determine_memory_layout(stream_memory_layout);
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to determine memory layout for stream " << stream->get_stream_name() << " of receiver " << get_index() << std::endl;
            return rc;
        }
        std::pair<size_t, size_t> aligned_buffer_sizes =
            m_memory_utils.align_buffer_sizes({stream_memory_layout.buffer_sizes.header_buffer_size,
                                               stream_memory_layout.buffer_sizes.payload_buffer_size});
        StreamBufferMemorySizes buffer_sizes;
        buffer_sizes.header_buffer_size = aligned_buffer_sizes.first;
        buffer_sizes.payload_buffer_size = aligned_buffer_sizes.second;
        buffer_sizes.auxiliary_buffer_size = stream_memory_layout.buffer_sizes.auxiliary_buffer_size;
        m_buffer_sizes.push_back(buffer_sizes);

        header_memory_size += buffer_sizes.header_buffer_size;
        payload_memory_size += buffer_sizes.payload_buffer_size;
        auxiliary_memory_size += buffer_sizes.auxiliary_buffer_size;
    }

    m_header_total_memory_size = header_memory_size;
    m_payload_total_memory_size = payload_memory_size;
    m_auxiliary_total_memory_size = auxiliary_memory_size;

    return ReturnStatus::success;
}

ReturnStatus ReceiverIONodeBase::determine_memory_layout(StreamMemoryLayoutRequest& memory_layout_request) const
{
    if (!is_memory_layout_initialized()) {
        std::cerr << "Memory layout was not initialized for receiver " << get_index() << std::endl;
        return ReturnStatus::failure;
    }
    auto& buffer_sizes = memory_layout_request.buffer_sizes;
    buffer_sizes.header_buffer_size = m_header_total_memory_size;
    buffer_sizes.payload_buffer_size = m_payload_total_memory_size;
    buffer_sizes.auxiliary_buffer_size = m_auxiliary_total_memory_size;

    return ReturnStatus::success;
}

ReturnStatus ReceiverIONodeBase::apply_memory_layout(const StreamMemoryLayoutResponse& memory_layout_response)
{
    ReturnStatus rc = validate_memory_layout(memory_layout_response);
    if (rc != ReturnStatus::success) {
        std::cerr << "Invalid memory layout provided for receiver " << get_index() << std::endl;
        return rc;
    }

    const auto& provided_memory_layout = memory_layout_response.memory_layout;
    size_t header_offset = 0;
    size_t payload_offset = 0;
    size_t auxiliary_offset = 0;

    for (size_t i = 0; i < m_streams.size(); ++i) {
        auto& stream = m_streams[i];
        auto& stream_buffer_sizes = m_buffer_sizes[i];

        rc = apply_memory_layout_for_subcomponent(stream, provided_memory_layout,
            stream_buffer_sizes, header_offset, payload_offset, auxiliary_offset);
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to apply memory layout for stream " << stream->get_stream_name() << " of receiver " << get_index() << std::endl;
            return rc;
        }
    }
    return ReturnStatus::success;
}

ReturnStatus ReceiverIONodeBase::validate_memory_layout(const StreamMemoryLayoutResponse& memory_layout_response) const
{
    if (!is_memory_layout_initialized()) {
        std::cerr << "Memory layout was not initialized for receiver " << get_index() << std::endl;
        return ReturnStatus::failure;
    }

    const auto& provided_memory_layout = memory_layout_response.memory_layout;

    if ((m_header_total_memory_size > 0 && provided_memory_layout.header_memory_ptr == nullptr) ||
        provided_memory_layout.payload_memory_ptr == nullptr) {
        std::cerr << "Invalid memory layout provided" << std::endl;
        return ReturnStatus::failure;
    }

    if (provided_memory_layout.payload_memory_size < m_payload_total_memory_size ||
        provided_memory_layout.header_memory_size < m_header_total_memory_size ||
        (provided_memory_layout.auxiliary_memory_ptr && provided_memory_layout.auxiliary_memory_size < m_auxiliary_total_memory_size)) {
        std::cerr << "Insufficient memory size provided" << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

ReturnStatus ReceiverIONodeBase::apply_memory_layout_for_subcomponent(
    std::unique_ptr<IReceiveStream>& stream,
    const StreamMemoryLayout& provided_memory_layout,
    const StreamBufferMemorySizes& stream_buffer_sizes,
    size_t& header_offset, size_t& payload_offset, size_t& auxiliary_offset) const
{
    StreamMemoryLayoutResponse stream_memory_layout_response(provided_memory_layout.register_memory,
        provided_memory_layout.header_memory_keys.size());
    auto& stream_memory_layout = stream_memory_layout_response.memory_layout;

    if (provided_memory_layout.header_memory_ptr) {
        stream_memory_layout.header_memory_ptr =
            static_cast<uint8_t*>(provided_memory_layout.header_memory_ptr) + header_offset;
    } else {
        stream_memory_layout.header_memory_ptr = nullptr;
    }
    stream_memory_layout.payload_memory_ptr =
        static_cast<uint8_t*>(provided_memory_layout.payload_memory_ptr) + payload_offset;
    if (provided_memory_layout.auxiliary_memory_ptr) {
        stream_memory_layout.auxiliary_memory_ptr =
            static_cast<uint8_t*>(provided_memory_layout.auxiliary_memory_ptr) + auxiliary_offset;
    } else {
        stream_memory_layout.auxiliary_memory_ptr = nullptr;
    }

    stream_memory_layout.header_memory_size = stream_buffer_sizes.header_buffer_size;
    stream_memory_layout.payload_memory_size = stream_buffer_sizes.payload_buffer_size;
    stream_memory_layout.auxiliary_memory_size = stream_buffer_sizes.auxiliary_buffer_size;

    if (provided_memory_layout.register_memory) {
        std::copy(provided_memory_layout.header_memory_keys.begin(),
            provided_memory_layout.header_memory_keys.end(),
            stream_memory_layout.header_memory_keys.begin());
        std::copy(provided_memory_layout.payload_memory_keys.begin(),
            provided_memory_layout.payload_memory_keys.end(),
            stream_memory_layout.payload_memory_keys.begin());
    }

    header_offset += stream_buffer_sizes.header_buffer_size;
    payload_offset += stream_buffer_sizes.payload_buffer_size;
    auxiliary_offset += stream_buffer_sizes.auxiliary_buffer_size;

    ReturnStatus rc = stream->apply_memory_layout(stream_memory_layout_response);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to set memory layout for stream " << stream->get_stream_name() << " of receiver " << get_index() << std::endl;
        return rc;
    }
    return ReturnStatus::success;
}

void ReceiverIONodeBase::set_cpu_resources()
{
    set_current_thread_affinity(m_cpu_core_affinity);
    rt_set_thread_priority(RMAX_THREAD_PRIORITY_TIME_CRITICAL - 1);
}

ReturnStatus ReceiverIONodeBase::update_streams_runtime_parameters()
{
    ReturnStatus status = ReturnStatus::success;
    for (auto& stream : m_streams) {
        status = stream->apply_runtime_parameters();
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to update runtime parameters for stream " << stream->get_stream_name() << " of receiver " << get_index() << std::endl;
            return status;
        }
    }
    return ReturnStatus::success;
}

void ReceiverIONodeBase::print_statistics(
    std::ostream& out, const std::chrono::high_resolution_clock::duration& interval_duration) const
{
    for (const auto& stream : m_streams) {
        stream->print_statistics(out, interval_duration);
        stream->reset_statistics();
    }
}

ReceiveChunk& ReceiverIONodeBase::get_stream_chunk(size_t stream_index) const
{
    assert(stream_index < m_streams.size());
    assert(stream_index < m_stream_chunks.size());

    return *m_stream_chunks[stream_index];
}

std::unique_ptr<ReceiveChunk> ReceiverIONodeBase::create_chunk(size_t stream_index) const
{
    assert(stream_index < m_streams.size());
    return std::make_unique<ReceiveChunk>(m_streams[stream_index]->get_id(), m_app_settings.packet_app_header_size != 0);
}

ReturnStatus ReceiverIONodeBase::set_receive_data_consumer(size_t stream_index, std::unique_ptr<IReceiveDataConsumer> data_consumer)
{
    if (data_consumer == nullptr) {
        std::cerr << "Invalid data consumer" << std::endl;
        return ReturnStatus::failure;
    }
    if (stream_index >= m_streams.size()) {
        std::cerr << "Invalid stream index " << stream_index << std::endl;
        return ReturnStatus::failure;
    }
    m_data_consumers[stream_index] = std::move(data_consumer);
    return ReturnStatus::success;
}

ReturnStatus ReceiverIONodeBase::consume_chunk(std::unique_ptr<IReceiveDataConsumer>& data_consumer,
    const ReceiveChunk& chunk, IReceiveStream& stream)
{
    assert(data_consumer);

    size_t consumed_packets = 0;
    auto rc = data_consumer->consume_chunk(chunk, stream, consumed_packets);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to consume chunk" << std::endl;
    }

    assert(consumed_packets <= chunk.get_length());

    return rc;
}

ReturnStatus ReceiverIONodeBase::wait_first_packet()
{
    ReturnStatus rc = ReturnStatus::success;
    bool initialized = false;
    while (likely(!initialized && rc != ReturnStatus::failure && SignalHandler::get_received_signal() < 0)) {
        for (int i = 0; i < m_streams.size(); i++) {
            auto& stream = m_streams[i];
            auto& chunk = get_stream_chunk(i);
            auto& data_consumer = m_data_consumers[i];
            rc = stream->get_next_chunk(chunk);
            if (unlikely(rc != ReturnStatus::success)) {
                break;
            }
            if (chunk.get_length() > 0) {
                initialized = true;
                rc = consume_chunk(data_consumer, chunk, *(stream.get()));
                if (rc != ReturnStatus::success) {
                    std::cerr << "Error processing chunk of packets" << std::endl;
                }
                break;
            }
        }
    }
    return rc;
}

void ReceiverIONodeBase::operator()()
{
    assert(m_streams.size() > 0);
    assert(m_streams.size() == m_data_consumers.size());

    set_cpu_resources();

    ReturnStatus rc = create_streams();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to create receiver (" << m_index << ") streams" << std::endl;
        return;
    }

    rc = update_streams_runtime_parameters();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to update stream runtime parameters for receiver (" << m_index << ")" << std::endl;
        return;
    }

    rc = attach_flows();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to attach flows to receiver (" << m_index << ") streams" << std::endl;
        return;
    }

    print_parameters();

    rc = synchronous_start();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Error starting receive process" << std::endl;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    while (likely(rc != ReturnStatus::failure && SignalHandler::get_received_signal() < 0)) {
        for (int i = 0; i < m_streams.size(); i++) {
            auto& stream = m_streams[i];
            auto& chunk = get_stream_chunk(i);
            auto& data_consumer = m_data_consumers[i];
            rc = stream->get_next_chunk(chunk);
            if (unlikely(rc != ReturnStatus::success)) {
                break;
            }
            if (chunk.get_length() == 0) {
                continue;
            }
            rc = consume_chunk(data_consumer, chunk, *(stream.get()));
            if (rc != ReturnStatus::success) {
                std::cerr << "Error processing chunk of packets" << std::endl;
            }
        }
        if (m_print_interval_ms > 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = now - start_time;
            if (duration >= std::chrono::milliseconds{m_print_interval_ms}) {
                print_statistics(std::cout, duration);
                start_time = now;
            }
        }
        if (m_sleep_between_operations.count() > 0) {
            std::this_thread::sleep_for(m_sleep_between_operations);
        }
    }

    rc = detach_flows();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to detach receiver (" << m_index << ")  flows" << std::endl;
    }

    rc = destroy_streams();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to destroy receiver (" << m_index << ") streams" << std::endl;
        return;
    }
}
