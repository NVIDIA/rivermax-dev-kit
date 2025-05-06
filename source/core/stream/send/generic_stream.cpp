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
#include <chrono>
#include <thread>
#include <memory>

#include "rdk/services/utils/defs.h"
#include "rdk/core/stream/send/generic_stream.h"
#include "rdk/core/chunk/generic_chunk.h"
#include "rdk/core/flow/flow.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

constexpr size_t MEM_SUBBLOCKS = 2;

GenericStreamSettings::GenericStreamSettings(const FourTupleFlow& network_address, bool fixed_dest_addr,
        PacketPacingRate pp_rate, size_t num_of_requested_chunks, size_t num_of_packets_in_chunk,
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
    ReturnStatus status = initialize_memory_layout();
    if (status != ReturnStatus::success) {
        throw std::runtime_error("Failed to initialize memory layout");
    }
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
    return status;
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

ReturnStatus GenericSendStream::initialize_memory_layout()
{
    m_header_buffer_size = m_num_of_chunks * m_stream_settings.m_num_of_packets_in_chunk *
        m_stream_settings.m_packet_typical_app_header_size;
    m_payload_buffer_size = m_num_of_chunks * m_stream_settings.m_num_of_packets_in_chunk *
        m_stream_settings.m_packet_typical_payload_size;
    return ReturnStatus::success;
}

ReturnStatus GenericSendStream::determine_memory_layout(HeaderPayloadMemoryLayoutRequest& memory_layout_request) const
{
    memory_layout_request.header_payload_buffers_size =
        {get_header_memory_length(), get_payload_memory_length()};
    return ReturnStatus::success;
}

ReturnStatus GenericSendStream::apply_memory_layout(const HeaderPayloadMemoryLayoutResponse& memory_layout_response)
{
    ReturnStatus status = validate_memory_layout(memory_layout_response);
    if (status != ReturnStatus::success) {
        std::cerr << "Failed to validate memory layout" << std::endl;
        return status;
    }

    const auto& stream_memory_layout = memory_layout_response.memory_layout;
    if (stream_memory_layout.header_memory_size > 0) {
        rmx_mem_region header_mem_region;
        header_mem_region.addr = stream_memory_layout.header_memory_ptr;
        header_mem_region.length = stream_memory_layout.header_memory_size;
        header_mem_region.mkey = stream_memory_layout.header_memory_keys[0];
        rmx_mem_region payload_mem_region;
        payload_mem_region.addr = stream_memory_layout.payload_memory_ptr;
        payload_mem_region.length = stream_memory_layout.payload_memory_size;
        payload_mem_region.mkey = stream_memory_layout.payload_memory_keys[0];
        return initialize_chunks(header_mem_region, payload_mem_region);
    } else {
        rmx_mem_region mem_region;
        mem_region.addr = stream_memory_layout.payload_memory_ptr;
        mem_region.length = stream_memory_layout.payload_memory_size;
        mem_region.mkey = stream_memory_layout.payload_memory_keys[0];
        return initialize_chunks(mem_region);
    }
}

ReturnStatus GenericSendStream::validate_memory_layout(const HeaderPayloadMemoryLayoutResponse& memory_layout_response) const
{
    const auto& stream_memory_layout = memory_layout_response.memory_layout;
    if (!stream_memory_layout.register_memory) {
        std::cerr << "Memory is not registered" << std::endl;
        return ReturnStatus::failure;
    }

    if (stream_memory_layout.header_memory_keys.size() != 1 ||
    stream_memory_layout.payload_memory_keys.size() != 1) {
        std::cerr << "Invalid number of memory keys" << std::endl;
        return ReturnStatus::failure;
    }
    if (stream_memory_layout.header_memory_size > 0) {
        if (stream_memory_layout.header_memory_size < get_header_memory_length() ||
            stream_memory_layout.payload_memory_size < get_payload_memory_length()) {
            std::cerr << "Invalid buffer size" << std::endl;
            return ReturnStatus::failure;
        }
    } else {
        if (stream_memory_layout.payload_memory_size < get_memory_length()) {
            std::cerr << "Invalid buffer size" << std::endl;
            return ReturnStatus::failure;
        }
    }

    return ReturnStatus::success;
}

ReturnStatus GenericSendStream::initialize_chunks(const rmx_mem_region& mem_region)
{
    if (!m_stream_created) {
        std::cerr << "Failed to initialize chunks, the stream was not created" << std::endl;
        return ReturnStatus::failure;
    }
    /* Initialize chunks */
    size_t mem_offset = 0;
    size_t num_of_sub_blocks = (m_stream_settings.m_packet_typical_app_header_size != 0) ? 2 : 1;
    GenericPacket packet(num_of_sub_blocks);
    for (size_t index = 0; index < m_num_of_chunks; index++) {
        auto chunk = std::make_shared<GenericChunk>(
                m_stream_id, m_stream_settings.m_num_of_packets_in_chunk);
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
        m_chunks.emplace_back(std::move(chunk));
    }
    return ReturnStatus::success;
}

ReturnStatus GenericSendStream::initialize_chunks(const rmx_mem_region& header_region, const rmx_mem_region& payload_region)
{
    if (!m_stream_created) {
        std::cerr << "Failed to initialize chunks, the stream was not created" << std::endl;
        return ReturnStatus::failure;
    }

    /* Initialize chunks */
    size_t header_mem_offset = 0;
    size_t payload_mem_offset = 0;
    size_t num_of_sub_blocks = 2;
    GenericPacket packet(num_of_sub_blocks);
    for (size_t index = 0; index < m_num_of_chunks; index++) {
        auto chunk = std::make_shared<GenericChunk>(
                m_stream_id, m_stream_settings.m_num_of_packets_in_chunk);
        /* Initialize packets */
        for (size_t packet_idx = 0; packet_idx < m_stream_settings.m_num_of_packets_in_chunk; packet_idx++) {
            /* Initialize IO vector */
            packet[0].addr = reinterpret_cast<uint8_t*>(header_region.addr) + header_mem_offset;
            packet[0].length = m_stream_settings.m_packet_typical_app_header_size;
            packet[0].mkey = header_region.mkey;
            header_mem_offset += packet[0].length;
            packet[1].addr = reinterpret_cast<uint8_t*>(payload_region.addr) + payload_mem_offset;
            packet[1].length = m_stream_settings.m_packet_typical_payload_size;
            packet[1].mkey = payload_region.mkey;
            payload_mem_offset += packet[1].length;
            /* Only store packets here. They need to be registered by Rivermax after each get_next_chunk() */
            chunk->place_packet(packet_idx, packet);
        }
        m_chunks.emplace_back(std::move(chunk));
    }
    return ReturnStatus::success;
}

size_t GenericSendStream::get_memory_length() const
{
    return get_header_memory_length() + get_payload_memory_length();
}

size_t GenericSendStream::get_header_memory_length() const
{
    return m_header_buffer_size;
}

size_t GenericSendStream::get_payload_memory_length() const
{
    return m_payload_buffer_size;
}

ReturnStatus GenericSendStream::create_stream()
{
    rmx_status status = rmx_output_gen_create_stream(&m_stream_params, &m_stream_id);
    if (status != RMX_OK) {
        std::cerr << "Failed to create generic stream with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
    m_stream_created = true;

    print(std::cout);

    return ReturnStatus::success;
}

ReturnStatus GenericSendStream::destroy_stream()
{
    if (!m_stream_created) {
        std::cerr << "Failed to destroy stream, the stream was not created" << std::endl;
        return ReturnStatus::failure;
    }
    rmx_status status;

    do {
        status = rmx_output_gen_destroy_stream(m_stream_id);
        if (status == RMX_BUSY) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } while (status == RMX_BUSY);

    if (status != RMX_OK) {
        std::cerr << "Failed to destroy generic stream with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
    m_stream_created = false;
    return ReturnStatus::success;
}
