/*
 * Copyright © 2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "core/stream/receive/receive_stream.h"
#include "services/error_handling/error_handling.h"
#include "services/utils/defs.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

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
    ISingleStream(settings.m_local_addr),
    m_stream_settings(settings),
    m_data_stride_size(0),
    m_hdr_stride_size(0),
    m_buffer_elements(settings.m_capacity_in_packets),
    m_header_mem_block_id(settings.m_header_size ? 0 : 1),
    m_payload_mem_block_id(settings.m_header_size ? 1 : 0),
    m_header_block(nullptr), m_payload_block(nullptr)
{
    m_stream_settings.build(m_stream_settings, m_stream_params);
}

ReturnStatus ReceiveStream::create_stream()
{
    rmx_status status = rmx_input_create_stream(&m_stream_params, &m_stream_id);
    if (status != RMX_OK) {
        std::cerr << "Failed to create receive stream with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
    m_stream_id_set = true;
    return ReturnStatus::success;
}

ReturnStatus ReceiveStream::set_completion_moderation(size_t min_count, size_t max_count, int timeout_usec)
{
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

ReturnStatus ReceiveStream::attach_flow(const FourTupleFlow& flow)
{
    rmx_input_flow rx_flow;

    rmx_input_init_flow(&rx_flow);
    rmx_input_set_flow_local_addr(&rx_flow, &flow.get_destination_socket_address());
    rmx_input_set_flow_remote_addr(&rx_flow, &flow.get_source_socket_address());

    rmx_status status = rmx_input_attach_flow(m_stream_id, &rx_flow);
    if (status != RMX_OK) {
        std::cerr << "Failed to attach flow with status: " << status << std::endl;
        return ReturnStatus::failure;
    }

    m_flows.emplace(flow, std::move(rx_flow));

    return ReturnStatus::success;
}

ReturnStatus ReceiveStream::detach_flow(const FourTupleFlow& flow)
{
    auto it = m_flows.find(flow);
    if (it == m_flows.end()) {
        std::cerr << "Failed to detach flow, it doesn't exist" << std::endl;
        return ReturnStatus::failure;
    }

    rmx_input_flow& rx_flow = it->second;

    rmx_status status = rmx_input_detach_flow(m_stream_id, &rx_flow);
    if (status != RMX_OK && status != RMX_SIGNAL) {
        std::cerr << "Failed to detach flow with status: " << status << std::endl;
        return ReturnStatus::failure;
    }

    m_flows.erase(it);

    return ReturnStatus::success;
}

ReturnStatus ReceiveStream::destroy_stream()
{
    rmx_status status = rmx_input_destroy_stream(m_stream_id);
    if (status != RMX_OK) {
        std::cerr << "Failed to destroy receive stream with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
    m_stream_id_set = false;
    return ReturnStatus::success;
}

ReturnStatus ReceiveStream::query_buffer_size(size_t& header_buffer_size, size_t& payload_buffer_size)
{
    rmx_status status = RMX_OK;
    header_buffer_size = 0;
    payload_buffer_size = 0;

    status = rmx_input_determine_mem_layout(&m_stream_params);
    if (status != RMX_OK) {
        return ReturnStatus::failure;
    }

    m_data_stride_size = rmx_input_get_stride_size(&m_stream_params, m_payload_mem_block_id);
    if (is_header_data_split_on()) {
        m_hdr_stride_size = rmx_input_get_stride_size(&m_stream_params, m_header_mem_block_id);
    } else {
        m_hdr_stride_size = 0;
    }

    if (!m_payload_block) {
        m_payload_block = rmx_input_get_mem_block_buffer(&m_stream_params, m_payload_mem_block_id);
    }
    // m_hdr will be NULL if HDS is not in use
    if (is_header_data_split_on() && !m_header_block) {
        m_header_block = rmx_input_get_mem_block_buffer(&m_stream_params, m_header_mem_block_id);
    }

    m_buffer_elements = (uint32_t)rmx_input_get_mem_capacity_in_packets(&m_stream_params);

    payload_buffer_size = m_payload_block->length;
    if (m_header_block != nullptr) {
        header_buffer_size = m_header_block->length;
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
    ISingleStream::print(out);

    out << "| Number of flows: " << m_flows.size() << "\n"
        << "| Number of packets in memory: " << m_stream_settings.m_capacity_in_packets << "\n"
        << "| Packet's header size: " << m_stream_settings.m_header_size << "\n"
        << "| Packet's payload size: " << m_stream_settings.m_payload_size << "\n";
    for (const auto& item : m_flows) {
        const auto& flow = item.first;

        out << "| Flow: id " << flow.get_id()
            << ", source " << flow.get_source_ip() << ":" << flow.get_source_port()
            << ", destination " << flow.get_destination_ip() << ":" << flow.get_destination_port() << "\n";
    }
    out << "+**********************************************\n";

    return out;
}
