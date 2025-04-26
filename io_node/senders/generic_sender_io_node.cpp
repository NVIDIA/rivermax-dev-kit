/*
 * Copyright © 2017-2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#include <thread>
#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>
#include <cstring>

#include <rivermax_api.h>

#include "rt_threads.h"

#include "senders/generic_sender_io_node.h"
#include "api/rmax_apps_lib_api.h"

using namespace ral::io_node;
using namespace ral::lib::core;
using namespace ral::lib::services;

AppGenericSendStream::AppGenericSendStream(
    const FourTupleFlow& network_address, std::vector<TwoTupleFlow>& send_flows,
    pp_rate_t rate, size_t num_of_requested_chunks, size_t num_of_packets_in_chunk,
    uint16_t packet_typical_payload_size, uint16_t packet_typical_app_header_size) :
    GenericSendStream(
        network_address, rate, num_of_requested_chunks, num_of_packets_in_chunk,
        packet_typical_payload_size, packet_typical_app_header_size),
    m_send_flows(send_flows),
    m_next_flow_to_send(0)
{
}

std::ostream& AppGenericSendStream::print(std::ostream& out) const
{
    GenericSendStream::print(out);

    out << "| Number of flows: " << m_send_flows.size() << "\n"
        << "+**********************************************\n";

    return out;
}

TwoTupleFlow* AppGenericSendStream::get_next_flow_to_send()
{
    /**
     * In case there is one flow in the stream,
     * the stream will use @ref rmax_out_commit_chunk API call,
     * otherwise will use the @ref rmax_out_commit_chunk_to API call
     * in order to commit the data. This is due to limitation of sending
     * Unicast traffic with @ref rmax_out_commit_chunk_to API call.
     */
    if (m_send_flows.size() == 1) {
        return nullptr;
    } else {
        return &m_send_flows[m_next_flow_to_send];
    }
}

void AppGenericSendStream::set_next_flow_to_send()
{
    ++m_next_flow_to_send %= m_send_flows.size();
}

GenericSenderIONode::GenericSenderIONode(
    std::shared_ptr<AppSettings> app_settings,
    size_t index, size_t num_of_streams, int cpu_core_affinity,
    std::shared_ptr<MemoryUtils> mem_utils) :
    m_index(index),
    m_num_of_streams(num_of_streams),
    m_sleep_between_operations_us(app_settings->sleep_between_operations_us),
    m_print_parameters(app_settings->print_parameters),
    m_network_address(FourTupleFlow(index, app_settings->source_ip, app_settings->source_port,
        app_settings->destination_ip, app_settings->destination_port)),
    m_rate(app_settings->rate),
    m_num_of_chunks(app_settings->num_of_chunks),
    m_num_of_packets_in_chunk(app_settings->num_of_packets_in_chunk),
    m_packet_typical_payload_size(app_settings->packet_payload_size),
    m_packet_typical_app_header_size(app_settings->packet_app_header_size),
    m_use_checksum_header(app_settings->use_checksum_header),
    m_cpu_core_affinity(cpu_core_affinity),
    m_hw_queue_full_sleep_us(app_settings->hw_queue_full_sleep_us),
    m_buffer_writer(std::unique_ptr<IBufferWriter>(app_settings->use_checksum_header
        ? dynamic_cast<IBufferWriter*>(new ChecksumBufferWriter())
        : dynamic_cast<IBufferWriter*>(new GenericBufferWriter()))),
    m_mem_utils(mem_utils)
{
}

std::ostream& GenericSenderIONode::print(std::ostream& out) const
{
    out << "+#############################################\n"
        << "| Sender index: " << m_index << "\n"
        << "| Thread ID: 0x" << std::hex << std::this_thread::get_id() << std::dec << "\n"
        << "| CPU core affinity: " << m_cpu_core_affinity << "\n"
        << "| Number of streams in this thread: " << m_streams.size() << "\n"
        << "| Memory address: " << m_mem_block.mem_block.pointer << "\n"
        << "| Memory length: " << m_mem_block.mem_block.length << "[B]" << "\n"
        << "| Memory key: " << m_mem_block.mkey_id << "\n"
        << "+#############################################\n";
    return out;
}

void GenericSenderIONode::initialize_send_flows(const std::vector<TwoTupleFlow>& flows)
{
    std::unordered_map<size_t,size_t> flows_per_stream;

    flows_per_stream.reserve(flows.size());
    for (size_t flow = 0; flow < flows.size(); flow++) {
        flows_per_stream[flow % m_num_of_streams]++;
    }

    size_t flows_offset = 0;

    for (size_t strm_indx = 0; strm_indx < m_num_of_streams; strm_indx++) {
        m_flows_in_stream[strm_indx] = std::vector<TwoTupleFlow>(
            flows.begin() + flows_offset,
            flows.begin() + flows_offset + flows_per_stream[strm_indx]);
        flows_offset += flows_per_stream[strm_indx];
    }
}

void GenericSenderIONode::initialize_streams(size_t& flows_offset)
{
    for (size_t strm_indx = 0; strm_indx < m_num_of_streams; strm_indx++) {
        m_streams.push_back(std::unique_ptr<AppGenericSendStream>(new AppGenericSendStream(
            FourTupleFlow(
                strm_indx,
                m_network_address.get_source_ip(),
                m_network_address.get_source_port(),
                m_network_address.get_destination_ip(),
                m_network_address.get_destination_port() + static_cast<uint16_t>(flows_offset)),
            m_flows_in_stream[strm_indx],
            /**
             * The rate limit configuration below done, in order to get rate limit
             * fairness between multiple flows in one stream.
             */
            {
                static_cast<uint64_t>(m_rate.bps * m_flows_in_stream[strm_indx].size()),
                static_cast<uint32_t>(m_rate.max_burst_in_packets * m_flows_in_stream[strm_indx].size() * m_num_of_packets_in_chunk)
            },
            m_num_of_chunks,
            m_num_of_packets_in_chunk,
            m_packet_typical_payload_size,
            m_packet_typical_app_header_size)));
        flows_offset += m_flows_in_stream[strm_indx].size();
    }
}

size_t GenericSenderIONode::initialize_memory(void* pointer, rmax_mkey_id mkey)
{
    memset(&m_mem_block, 0, sizeof(m_mem_block));
    m_mem_block.mem_block.pointer = pointer;
    m_mem_block.mkey_id = mkey;
    for (auto& stream : m_streams) {
        m_mem_block.mem_block.length += stream->get_memory_length();
    }
    distribute_memory_for_streams();
    return m_mem_block.mem_block.length;
}

void GenericSenderIONode::distribute_memory_for_streams()
{
    byte_t* pointer = nullptr;
    rmax_mkey_id mkey = 0;
    size_t length = 0;
    size_t offset = 0;

    for (auto& stream : m_streams) {
        pointer = reinterpret_cast<byte_t*>(m_mem_block.mem_block.pointer) + offset;
        mkey = m_mem_block.mkey_id;
        length = stream->initialize_chunks(pointer, mkey);
        offset += length;
    }
}

void GenericSenderIONode::print_parameters()
{
    if (!m_print_parameters) {
        return;
    }

    std::stringstream sender_parameters;
    sender_parameters << this;
    for (auto& stream : m_streams) {
        sender_parameters << *stream;
    }
    std::cout << sender_parameters.str() << std::endl;
}

void GenericSenderIONode::operator()()
{
    set_cpu_resources();
    ReturnStatus rc = create_streams();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to create sender (" << m_index << ") streams" << std::endl;
        return;
    }
    print_parameters();
    prepare_buffers();

    size_t num_of_streams = m_streams.size();
    uint64_t commit_timestamp_ns = 0;
    rmax_commit_flags_t commit_flags = static_cast<rmax_commit_flags_t>(0);
    GenericChunk commit_chunk;

    while (likely(rc != ReturnStatus::failure && rc != ReturnStatus::signal_received &&
                  SignalHandler::get_received_signal() < 0)) {
        for (size_t chunk = 0; chunk < m_num_of_chunks; chunk++) {
            size_t stream_index = 0;
            do {
                auto& stream = m_streams[stream_index];
                stream->get_next_chunk(&commit_chunk);
                auto* flow = stream->get_next_flow_to_send();
                rc = stream->blocking_commit_chunk(commit_chunk, commit_timestamp_ns, commit_flags, flow);
                switch (rc) {
                    case ReturnStatus::hw_send_queue_full:
                        std::this_thread::sleep_for(std::chrono::microseconds(m_hw_queue_full_sleep_us));
                        break;
                    case ReturnStatus::failure:
                        std::cerr << "Failed to send chunk of stream (" << stream->get_id() << ")" << std::endl;
                        break;
                    case ReturnStatus::signal_received:
                        std::cerr << "Received signal when send chunk of stream (" << stream->get_id() << ")" << std::endl;
                        break;
                        /* fall through */
                    default:
                        stream->set_next_flow_to_send();
                        ++stream_index;
                        break;
                }
            } while (likely(rc != ReturnStatus::failure && rc != ReturnStatus::signal_received) && stream_index < num_of_streams);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(m_sleep_between_operations_us));
    }

    rc = destroy_streams();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to destroy sender (" << m_index << ") streams" << std::endl;
        return;
    }
}

ReturnStatus GenericSenderIONode::create_streams()
{
    ReturnStatus rc;

    for (auto& stream : m_streams) {
        rc = stream->create_stream();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to create stream (" << stream->get_id() << ")" << std::endl;
            return rc;
        }
    }

    return ReturnStatus::success;
}

ReturnStatus GenericSenderIONode::destroy_streams()
{
    ReturnStatus rc;

    for (auto& stream : m_streams) {
        rc = stream->destroy_stream();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to destroy stream (" << stream->get_id() << ")" << std::endl;
            return rc;
        }
    }

    return ReturnStatus::success;
}

void GenericSenderIONode::set_cpu_resources()
{
    memset(&m_cpu_affinity_mask, 0, sizeof(m_cpu_affinity_mask));
    if (m_cpu_core_affinity != CPU_NONE) {
        RMAX_CPU_SET(m_cpu_core_affinity, &m_cpu_affinity_mask);
    }
    rt_set_thread_affinity(&m_cpu_affinity_mask);
    rt_set_thread_priority(RMAX_THREAD_PRIORITY_TIME_CRITICAL - 1);
}

inline void GenericSenderIONode::prepare_buffers()
{
    for (auto& stream : m_streams) {
        for (size_t chunk_index = 0; chunk_index < stream->get_num_of_chunks(); chunk_index++) {
            auto chunk = dynamic_cast<GenericSendStream*>(stream.get())->get_chunk(chunk_index);
            m_buffer_writer->write_buffer(*chunk, m_mem_utils);
        }
    }
}
