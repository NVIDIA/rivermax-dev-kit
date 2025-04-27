/*
 * Copyright © 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
#include <vector>
#include <unordered_map>

#include "core/stream/send/generic_stream.h"
#include "core/chunk/generic_chunk.h"
#include "core/flow/flow.h"
#include "services/error_handling/error_handling.h"
#include "services/utils/defs.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

constexpr size_t MEM_SUBBLOCKS = 2;

GenericStreamSettings::GenericStreamSettings(const FourTupleFlow& network_address, bool fixed_dest_addr,
        pp_rate_t pp_rate, size_t num_of_requested_chunks, size_t num_of_packets_in_chunk,
        uint16_t packet_typical_payload_size, uint16_t packet_typical_app_header_size) :
    IStreamSettings(s_build_steps),
    m_network_address(network_address),
    m_fixed_dest_addr(fixed_dest_addr),
    m_pp_rate(pp_rate),
    m_num_of_requested_chunks(num_of_requested_chunks),
    m_num_of_packets_in_chunk(num_of_packets_in_chunk),
    m_packet_typical_payload_size(packet_typical_payload_size),
    m_packet_typical_app_header_size(packet_typical_app_header_size)
{
}

IStreamSettings<GenericStreamSettings, rmx_output_gen_stream_params>::SetterSequence GenericStreamSettings::s_build_steps{
    &GenericStreamSettings::stream_param_init,
    &GenericStreamSettings::stream_param_set_local_addr,
    &GenericStreamSettings::stream_param_set_remote_addr,
    &GenericStreamSettings::stream_param_set_rate,
    &GenericStreamSettings::stream_param_set_chunk_size,
};

void GenericStreamSettings::stream_param_init(rmx_output_gen_stream_params& descr)
{
    rmx_output_gen_init_stream(&descr);
}

void GenericStreamSettings::stream_param_set_local_addr(rmx_output_gen_stream_params& descr)
{
    rmx_output_gen_set_local_addr(&descr, &m_network_address.get_source_socket_address());
}

void GenericStreamSettings::stream_param_set_remote_addr(rmx_output_gen_stream_params& descr)
{
    if (m_fixed_dest_addr) {
        rmx_output_gen_set_remote_addr(&descr, &m_network_address.get_destination_socket_address());
    }
}

void GenericStreamSettings::stream_param_set_rate(rmx_output_gen_stream_params& descr)
{
    if (m_pp_rate.bps != 0) {
        rmx_output_gen_rate rate;
        rmx_output_gen_init_rate(&rate, m_pp_rate.bps);
        rmx_output_gen_set_rate_max_burst(&rate, m_pp_rate.max_burst_in_packets);
        rmx_output_gen_set_rate_typical_packet_size(&rate, m_packet_typical_payload_size);
        rmx_output_gen_set_rate(&descr, &rate);
    }
}

void GenericStreamSettings::stream_param_set_chunk_size(rmx_output_gen_stream_params& descr)
{
    rmx_output_gen_set_packets_per_chunk(&descr, m_num_of_packets_in_chunk);
}

GenericSendStream::GenericSendStream(const GenericStreamSettings& settings) :
    ISendStream(settings.m_network_address.get_source_flow()),
    m_stream_settings(settings),
    m_next_chunk_to_send_index(0)
{
    m_stream_settings.build(m_stream_settings, m_stream_params);

    m_num_of_chunks = m_stream_settings.m_num_of_requested_chunks;
}

std::ostream& GenericSendStream::print(std::ostream& out) const
{
    ISendStream::print(out);

    out << "| Rate limit bps: " << m_stream_settings.m_pp_rate.bps << "\n"
        << "| Rate limit max burst in packets: " << m_stream_settings.m_pp_rate.max_burst_in_packets << "\n"
        << "| Memory length: " << get_memory_length() << "[B]" << "\n"
        << "| Number of user requested chunks: " << m_stream_settings.m_num_of_requested_chunks << "\n"
        << "| Number of application chunks: " << m_num_of_chunks << "\n"
        << "| Number of packets in chunk: " << m_stream_settings.m_num_of_packets_in_chunk << "\n"
        << "| Packet's payload size: " << m_stream_settings.m_packet_typical_payload_size << "\n"
        << "+**********************************************\n";

    return out;
}

ReturnStatus GenericSendStream::get_next_chunk(std::shared_ptr<GenericChunk>& chunk)
{
    chunk = m_chunks[m_next_chunk_to_send_index];
    ReturnStatus status = chunk->get_next_chunk();
    if (status == ReturnStatus::success) {
        m_next_chunk_to_send_index++;
        m_next_chunk_to_send_index %= m_num_of_chunks;
    }
    return ReturnStatus::failure;
}

ReturnStatus GenericSendStream::blocking_get_next_chunk(std::shared_ptr<GenericChunk>& chunk, size_t retries)
{
    ReturnStatus status;

    do {
        status = get_next_chunk(chunk);
    } while (unlikely(status == ReturnStatus::no_free_chunks && retries--));

    return status;
}

ReturnStatus GenericSendStream::commit_chunk(std::shared_ptr<GenericChunk> chunk, uint64_t time)
{
    return chunk->commit_chunk(time);
}

ReturnStatus GenericSendStream::blocking_commit_chunk(std::shared_ptr<GenericChunk> chunk, uint64_t time, size_t retries)
{
    ReturnStatus status;

    do {
        status = commit_chunk(chunk, time);
    } while (unlikely(status == ReturnStatus::hw_send_queue_full && retries--));

    return status;
}

void GenericSendStream::initialize_chunks(const rmx_mem_region& mem_region)
{
    /* Initialize chunks */
    size_t mem_offset = 0;
    size_t num_of_sub_blocks = (m_stream_settings.m_packet_typical_app_header_size != 0) ? 2 : 1;
    GenericPacket packet(num_of_sub_blocks);
    for (size_t index = 0; index < m_num_of_chunks; index++) {
        GenericChunk* chunk = new GenericChunk(m_stream_id, m_stream_settings.m_num_of_packets_in_chunk);
        /* Initialize packets */
        for (size_t packet_idx = 0; packet_idx < m_stream_settings.m_num_of_packets_in_chunk; packet_idx++) {
            /* Initialize IO vector */
            for (size_t sub_block_idx = 0; sub_block_idx < num_of_sub_blocks; sub_block_idx++) {
                packet[sub_block_idx].addr = reinterpret_cast<uint8_t*>(mem_region.addr) + mem_offset;
                packet[sub_block_idx].length = (sub_block_idx == num_of_sub_blocks - 1) ?
                                                     m_stream_settings.m_packet_typical_payload_size :
                                                     m_stream_settings.m_packet_typical_app_header_size;
                packet[sub_block_idx].mkey = mem_region.mkey;
                mem_offset += packet[sub_block_idx].length;
            }
            /* Only store packets here. They need to be registered by Rivermax after each get_next_chunk() */
            chunk->place_packet(packet_idx, packet);
        }
        m_chunks.emplace_back(chunk);
    }
}

size_t GenericSendStream::get_memory_length() const
{
    return m_num_of_chunks * m_stream_settings.m_num_of_packets_in_chunk * MEM_SUBBLOCKS *
           (m_stream_settings.m_packet_typical_payload_size +
           m_stream_settings.m_packet_typical_app_header_size);
}

ReturnStatus GenericSendStream::create_stream()
{
    rmx_status status = rmx_output_gen_create_stream(&m_stream_params, &m_stream_id);
    if (status != RMX_OK) {
        std::cerr << "Failed to create generic stream with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
    m_stream_id_set = true;

    print(std::cout);

    return ReturnStatus::success;
}

ReturnStatus GenericSendStream::destroy_stream()
{
    rmx_status status;

    do {
        status = rmx_output_gen_destroy_stream(m_stream_id);
    } while (status == RMX_BUSY);

    if (status != RMX_OK) {
        std::cerr << "Failed to destroy generic stream with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
    m_stream_id_set = false;
    return ReturnStatus::success;
}
