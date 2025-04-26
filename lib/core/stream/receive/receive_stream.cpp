/*
 * Copyright © 2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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

ReceiveStream::ReceiveStream(rmax_in_stream_type rx_type, const TwoTupleFlow& local_addr,
        const receive_stream_settings_t& settings) :
    ISingleStream(local_addr),
    m_rx_type(rx_type),
    m_settings(settings),
    m_timestamp_format(RMAX_PACKET_TIMESTAMP_RAW_NANO)
{
    memset(&m_buffer_attr, 0, sizeof(m_buffer_attr));
    memset(&m_header_block, 0, sizeof(m_header_block));
    memset(&m_payload_block, 0, sizeof(m_payload_block));

    m_buffer_attr.num_of_elements = m_settings.num_of_packets_in_chunk;
    m_buffer_attr.attr_flags = m_settings.buffer_attr_flags;

    if (m_settings.packet_app_header_size > 0) {
        m_header_block.ptr = nullptr;
        m_header_block.min_size = m_settings.packet_app_header_size;
        m_header_block.max_size = m_settings.packet_app_header_size;
        m_buffer_attr.hdr = &m_header_block;
    }

    m_payload_block.ptr = nullptr;
    m_payload_block.min_size = m_settings.packet_payload_size;
    m_payload_block.max_size = m_settings.packet_payload_size;
    m_buffer_attr.data = &m_payload_block;
}

ReturnStatus ReceiveStream::create_stream()
{
    rmax_status_t status = rmax_in_create_stream(m_rx_type,
            reinterpret_cast<sockaddr_in*>(&m_local_address.get_socket_address()),
            &m_buffer_attr,
            m_timestamp_format,
            m_settings.stream_flags,
            &m_stream_id);
    if (status != RMAX_OK) {
        std::cerr << "Failed to create receive stream with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

ReturnStatus ReceiveStream::attach_flow(const FourTupleFlow& flow)
{
    rmax_in_flow_attr rmax_flow;

    memset(&rmax_flow, 0, sizeof(rmax_flow));
    std::memcpy(&rmax_flow.local_addr, &flow.get_destination_socket_address(), sizeof(sockaddr_in));
    std::memcpy(&rmax_flow.remote_addr, &flow.get_source_socket_address(), sizeof(sockaddr_in));

    rmax_status_t status = rmax_in_attach_flow(m_stream_id, &rmax_flow);
    if (status != RMAX_OK) {
        std::cerr << "Failed to attach flow with status: " << status << std::endl;
        return ReturnStatus::failure;
    }

    m_flows.emplace(flow, std::move(rmax_flow));

    return ReturnStatus::success;
}

ReturnStatus ReceiveStream::detach_flow(const FourTupleFlow& flow)
{
    auto it = m_flows.find(flow);
    if (it == m_flows.end()) {
        std::cerr << "Failed to detach flow, it doesn't exist" << std::endl;
        return ReturnStatus::failure;
    }

    rmax_in_flow_attr& rmax_flow = it->second;

    rmax_status_t status = rmax_in_detach_flow(m_stream_id, &rmax_flow);
    if (status != RMAX_OK && status != RMAX_SIGNAL) {
        std::cerr << "Failed to detach flow with status: " << status << std::endl;
        return ReturnStatus::failure;
    }

    m_flows.erase(it);

    return ReturnStatus::success;
}

ReturnStatus ReceiveStream::destroy_stream()
{
    rmax_status_t status = rmax_in_destroy_stream(m_stream_id);
    if (status != RMAX_OK) {
        std::cerr << "Failed to destroy receive stream with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

ReturnStatus ReceiveStream::query_buffer_size(size_t& header_size, size_t& payload_size)
{
    rmax_status_t status = rmax_in_query_buffer_size(m_rx_type,
            reinterpret_cast<sockaddr_in*>(&m_local_address.get_socket_address()),
            &m_buffer_attr, &payload_size, &header_size);
    if (status != RMAX_OK) {
        std::cerr << "Buffer size query failed with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

void ReceiveStream::set_buffers(void* header_ptr, void* payload_ptr)
{
    m_header_block.ptr = header_ptr;
    m_payload_block.ptr = payload_ptr;
}

void ReceiveStream::set_memory_keys(rmax_mkey_id header_mkey, rmax_mkey_id payload_mkey)
{
    m_header_block.mkey = header_mkey;
    m_payload_block.mkey = payload_mkey;
    if (m_settings.packet_app_header_size > 0) {
        m_buffer_attr.attr_flags |= RMAX_IN_BUFFER_ATTR_BUFFER_APP_HDR_MKEY_IS_SET;
    }
    m_buffer_attr.attr_flags |= RMAX_IN_BUFFER_ATTR_BUFFER_DATA_MKEY_IS_SET;
}

ReturnStatus ReceiveStream::get_next_chunk(ReceiveChunk* chunk)
{
    if (!chunk) {
        std::cerr << "Invalid parameter: chunk must be non-NULL" << std::endl;
        return ReturnStatus::failure;
    }
    rmax_status_t status = rmax_in_get_next_chunk(m_stream_id,
        m_settings.min_chunk_size, m_settings.max_chunk_size,
        m_settings.timeout_us, 0, chunk->get_completion());
    switch (status) {
    case RMAX_OK:
        return ReturnStatus::success;
    case RMAX_SIGNAL:
        return ReturnStatus::signal_received;
    default:
        std::cerr << "Failed to get next chunk of stream " << m_stream_id << " with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
}

std::ostream& ReceiveStream::print(std::ostream& out) const
{
    ISingleStream::print(out);

    out << "| Number of flows: " << m_flows.size() << "\n"
        << "| Number of packets in chunk: " << m_settings.num_of_packets_in_chunk << "\n"
        << "| Packet's header size: " << m_settings.packet_app_header_size << "\n"
        << "| Packet's payload size: " << m_settings.packet_payload_size << "\n";
    for (const auto& item : m_flows) {
        const auto& flow = item.first;

        out << "| Flow: id " << flow.get_id()
            << ", source " << flow.get_source_ip() << ":" << flow.get_source_port()
            << ", destination " << flow.get_destination_ip() << ":" << flow.get_destination_port() << "\n";
    }
    out << "+**********************************************\n";

    return out;
}
