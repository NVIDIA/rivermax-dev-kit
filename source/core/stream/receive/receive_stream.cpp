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

#include <cstring>
#include <iostream>
#include <stdexcept>

#include "rdk/core/stream/receive/receive_stream.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

ReceiveStreamSettings::ReceiveStreamSettings(const TwoTupleFlow& local_addr,
        rmx_input_stream_params_type rx_type,
        rmx_input_timestamp_format ts_format,
        const std::unordered_set<rmx_input_option>& options,
        size_t capacity_in_packets, size_t payload_size, size_t header_size) :
    IStreamSettings(s_build_steps),
    m_local_addr(local_addr),
    m_rx_type(rx_type),
    m_ts_format(ts_format),
    m_options(options),
    m_capacity_in_packets(capacity_in_packets),
    m_payload_size(payload_size),
    m_header_size(header_size)
{
}

IStreamSettings<ReceiveStreamSettings, rmx_input_stream_params>::SetterSequence ReceiveStreamSettings::s_build_steps{
    &ReceiveStreamSettings::stream_param_init,
    &ReceiveStreamSettings::stream_param_set_nic_address,
    &ReceiveStreamSettings::stream_param_set_capacity,
    &ReceiveStreamSettings::stream_param_set_packet_size,
    &ReceiveStreamSettings::stream_param_set_ts_format,
    &ReceiveStreamSettings::stream_param_set_input_options,
};

void ReceiveStreamSettings::stream_param_init(rmx_input_stream_params& descr)
{
    rmx_input_init_stream(&descr, m_rx_type);
}

void ReceiveStreamSettings::stream_param_set_nic_address(rmx_input_stream_params& descr)
{
    rmx_input_set_stream_nic_address(&descr, &m_local_addr.get_socket_address());
}

void ReceiveStreamSettings::stream_param_set_capacity(rmx_input_stream_params& descr)
{
    rmx_input_set_mem_capacity_in_packets(&descr, m_capacity_in_packets);
}

void ReceiveStreamSettings::stream_param_set_packet_size(rmx_input_stream_params& descr)
{
    if (m_header_size == 0) {
        rmx_input_set_mem_sub_block_count(&descr, 1 /*without HDS*/);
        rmx_input_set_entry_size_range(&descr, 0, m_payload_size, m_payload_size);
    } else {
        rmx_input_set_mem_sub_block_count(&descr, 2 /*HDS*/);
        rmx_input_set_entry_size_range(&descr, 0, m_header_size, m_header_size);
        rmx_input_set_entry_size_range(&descr, 1, m_payload_size, m_payload_size);
    }
}

void ReceiveStreamSettings::stream_param_set_ts_format(rmx_input_stream_params& descr)
{
    rmx_input_set_timestamp_format(&descr, m_ts_format);
}

void ReceiveStreamSettings::stream_param_set_input_options(rmx_input_stream_params& descr)
{
    for(auto option : m_options) {
        rmx_input_enable_stream_option(&descr, option);
    }
}

ReceiveStream::ReceiveStream(const ReceiveStreamSettings& settings) :
    ISingleReceiveStream(settings.m_local_addr),
    m_stream_settings(settings),
    m_data_stride_size(0),
    m_header_stride_size(0),
    m_buffer_elements(settings.m_capacity_in_packets),
    m_header_mem_block_id(settings.m_header_size ? 0 : 1),
    m_payload_mem_block_id(settings.m_header_size ? 1 : 0),
    m_header_block(nullptr), m_payload_block(nullptr)
{
    m_stream_settings.build(m_stream_settings, m_stream_params);
    ReturnStatus status = initialize_memory_layout();
    if (status != ReturnStatus::success) {
        throw std::runtime_error("Failed to initialize memory layout");
    }
}

ReturnStatus ReceiveStream::create_stream()
{
    rmx_status status = rmx_input_create_stream(&m_stream_params, &m_stream_id);
    if (status != RMX_OK) {
        std::cerr << "Failed to create receive stream with status: " << status << std::endl;
        return ReturnStatus::failure;
    }

    m_stream_created = true;
    return ReturnStatus::success;
}

ReturnStatus ReceiveStream::set_completion_moderation(size_t min_count, size_t max_count, int timeout_usec)
{
    if (!m_stream_created) {
        std::cerr << "Failed to set completion moderation, the stream was not created" << std::endl;
        return ReturnStatus::failure;
    }

    rmx_status status = rmx_input_set_completion_moderation(m_stream_id, min_count, max_count, timeout_usec);
    if (status != RMX_OK) {
        std::cerr << "Failed to set expected packets count for stream: " << m_stream_id << ", with status: "
            << status << std::endl;
            return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus ReceiveStream::get_next_chunk(ReceiveChunk& chunk)
{
    return chunk.get_next_chunk();
}

ReturnStatus ReceiveStream::attach_flow(ReceiveFlow& flow)
{
    if (!m_stream_created) {
        std::cerr << "Failed to attach flow, the stream was not created" << std::endl;
        return ReturnStatus::failure;
    }

    auto& flow_descriptor = flow.init_descriptor();
    rmx_status status = rmx_input_attach_flow(m_stream_id, &flow_descriptor);
    if (status != RMX_OK) {
        std::cerr << "Failed to attach flow with status: " << status << std::endl;
        return ReturnStatus::failure;
    }

    m_flows.emplace(flow);

    return ReturnStatus::success;
}

ReturnStatus ReceiveStream::detach_flow(ReceiveFlow& flow)
{
    if (!m_stream_created) {
        std::cerr << "Failed to detach flow, the stream was not created" << std::endl;
        return ReturnStatus::failure;
    }

    auto it = m_flows.find(flow);
    if (it == m_flows.end()) {
        std::cerr << "Failed to detach flow, it doesn't exist" << std::endl;
        return ReturnStatus::failure;
    }

    auto& flow_descriptor = it->get_descriptor();
    rmx_status status = rmx_input_detach_flow(m_stream_id, &flow_descriptor);
    if (status != RMX_OK && status != RMX_SIGNAL) {
        std::cerr << "Failed to detach flow with status: " << status << std::endl;
        return ReturnStatus::failure;
    }

    m_flows.erase(it);

    return ReturnStatus::success;
}

ReturnStatus ReceiveStream::detach_flows()
{
    if (!m_stream_created) {
        std::cerr << "Failed to detach flows, the stream was not created" << std::endl;
        return ReturnStatus::failure;
    }

    int failed = 0;
    for (auto it = m_flows.begin(); it != m_flows.end(); ) {
        auto& flow_descriptor = it->get_descriptor();
        rmx_status status = rmx_input_detach_flow(m_stream_id, &flow_descriptor);
        if (status != RMX_OK && status != RMX_SIGNAL) {
            std::cerr << "Failed to detach flow from stream " << m_stream_id << " with status: " << status << std::endl;
            failed++;
        }
        it = m_flows.erase(it);
    }

    return failed > 0 ? ReturnStatus::failure : ReturnStatus::success;
}

ReturnStatus ReceiveStream::destroy_stream()
{
    rmx_status status = rmx_input_destroy_stream(m_stream_id);
    if (status != RMX_OK) {
        std::cerr << "Failed to destroy receive stream with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
    m_stream_created = false;
    return ReturnStatus::success;
}

ReturnStatus ReceiveStream::initialize_memory_layout()
{
    rmx_status status = rmx_input_determine_mem_layout(&m_stream_params);
    if (status != RMX_OK) {
        return ReturnStatus::failure;
    }

    m_data_stride_size = rmx_input_get_stride_size(&m_stream_params, m_payload_mem_block_id);
    if (is_header_data_split_on()) {
        m_header_stride_size = rmx_input_get_stride_size(&m_stream_params, m_header_mem_block_id);
    } else {
        m_header_stride_size = 0;
    }

    if (!m_payload_block) {
        m_payload_block = rmx_input_get_mem_block_buffer(&m_stream_params, m_payload_mem_block_id);
    }
    // m_header will be NULL if HDS is not in use
    if (is_header_data_split_on() && !m_header_block) {
        m_header_block = rmx_input_get_mem_block_buffer(&m_stream_params, m_header_mem_block_id);
    }

    m_buffer_elements = (uint32_t)rmx_input_get_mem_capacity_in_packets(&m_stream_params);

    return ReturnStatus::success;
}

ReturnStatus ReceiveStream::query_buffer_size(size_t& header_buffer_size, size_t& payload_buffer_size) const
{
    header_buffer_size = 0;
    payload_buffer_size = 0;

    if (!m_payload_block) {
        std::cerr << "Failed to query buffer size, Payload block is not initialized" << std::endl;
        return ReturnStatus::failure;
    }

    payload_buffer_size = m_payload_block->length;
    if (m_header_block != nullptr) {
        header_buffer_size = m_header_block->length;
    }

    return ReturnStatus::success;
}

ReturnStatus ReceiveStream::determine_memory_layout_helper(size_t& header_buffer_size, size_t& payload_buffer_size) const
{
    header_buffer_size = 0;
    payload_buffer_size = 0;

    if (!m_payload_block) {
        std::cerr << "Failed to query buffer size, Payload block is not initialized" << std::endl;
        return ReturnStatus::failure;
    }

    payload_buffer_size = m_payload_block->length;
    if (m_header_block != nullptr) {
        header_buffer_size = m_header_block->length;
    }

    return ReturnStatus::success;
}

ReturnStatus ReceiveStream::validate_memory_layout(const HeaderPayloadMemoryLayoutResponse& memory_layout_response) const
{
    const auto& stream_memory_layout = memory_layout_response.memory_layout;

    if (stream_memory_layout.register_memory &&
        (stream_memory_layout.header_memory_keys.size() != 1 ||
         stream_memory_layout.payload_memory_keys.size() != 1)) {
        std::cerr << "Invalid number of memory keys" << std::endl;
        return ReturnStatus::failure;
    }

    size_t header_buffer_size = 0;
    size_t payload_buffer_size = 0;

    ReturnStatus status = determine_memory_layout_helper(header_buffer_size, payload_buffer_size);
    if (status != ReturnStatus::success) {
        std::cerr << "Failed to query buffer size" << std::endl;
        return status;
    }

    if (stream_memory_layout.header_memory_size < header_buffer_size ||
        stream_memory_layout.payload_memory_size < payload_buffer_size) {
        std::cerr << "Invalid buffer size" << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus ReceiveStream::determine_memory_layout(HeaderPayloadMemoryLayoutRequest& memory_layout_request) const
{
    size_t header_buffer_size = 0;
    size_t payload_buffer_size = 0;

    ReturnStatus status = determine_memory_layout_helper(header_buffer_size, payload_buffer_size);
    if (status != ReturnStatus::success) {
        std::cerr << "Failed to determine memory layout" << std::endl;
        return status;
    }

    memory_layout_request.header_payload_buffers_size = {header_buffer_size, payload_buffer_size};

    return ReturnStatus::success;
}

ReturnStatus ReceiveStream::apply_memory_layout(const HeaderPayloadMemoryLayoutResponse& memory_layout_response)
{
    const auto& stream_memory_layout = memory_layout_response.memory_layout;
    ReturnStatus status = validate_memory_layout(memory_layout_response);
    if (status != ReturnStatus::success) {
        std::cerr << "Failed to validate memory layout" << std::endl;
        return status;
    }

    set_buffers(stream_memory_layout.header_memory_ptr, stream_memory_layout.payload_memory_ptr);

    if (stream_memory_layout.register_memory) {
        set_memory_keys(stream_memory_layout.header_memory_keys[0], stream_memory_layout.payload_memory_keys[0]);
    }

    return ReturnStatus::success;
}

void ReceiveStream::set_buffers(void* header_ptr, void* payload_ptr)
{
    if (m_header_block != nullptr) {
        m_header_block->addr = header_ptr;
    }
    m_payload_block->addr = payload_ptr;
}

void ReceiveStream::set_memory_keys(rmx_mkey_id header_mkey, rmx_mkey_id payload_mkey)
{
    if (m_header_block != nullptr) {
        m_header_block->mkey = header_mkey;
    }
    m_payload_block->mkey = payload_mkey;
}

std::ostream& ReceiveStream::print(std::ostream& out) const
{
    ISingleReceiveStream::print(out);

    out << "| Number of flows: " << m_flows.size() << "\n"
        << "| Number of packets in memory: " << m_stream_settings.m_capacity_in_packets << "\n"
        << "| Packet's header size: " << m_stream_settings.m_header_size << "\n"
        << "| Packet's payload size: " << m_stream_settings.m_payload_size << "\n";
    for (const auto& flow : m_flows) {
        out << "| Flow: id " << flow.get_id()
            << ", source " << flow.get_source_ip() << ":" << flow.get_source_port()
            << ", destination " << flow.get_destination_ip() << ":" << flow.get_destination_port() << "\n";
    }
    out << "+**********************************************\n";

    return out;
}
