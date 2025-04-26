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

#include <algorithm>
#include <cstring>

#include "core/stream/send/generic_stream.h"
#include "core/chunk/generic_chunk.h"
#include "core/flow/flow.h"
#include "services/error_handling/error_handling.h"
#include "services/utils/defs.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

GenericSendStream::GenericSendStream(
    const FourTupleFlow& network_address,
    pp_rate_t rate, size_t num_of_requested_chunks,
    size_t num_of_packets_in_chunk, uint16_t packet_typical_payload_size,
    uint16_t packet_typical_app_header_size) :
    ISendStream(network_address.get_source_flow()),
    m_destination_flow(network_address.get_destination_flow()),
    m_next_chunk_to_send_index(0),
    m_num_of_requested_chunks(num_of_requested_chunks),
    m_num_of_packets_in_chunk(num_of_packets_in_chunk),
    m_packet_payload_size(packet_typical_payload_size),
    m_packet_app_header_size(packet_typical_app_header_size),
    m_rate(rate)
{
    initialize_rmax_stream_parameters();
}

void GenericSendStream::initialize_rmax_stream_parameters()
{
    uint64_t flags = 0;

    memset(&m_rmax_parameters, 0, sizeof(m_rmax_parameters));
    m_rmax_parameters.local_addr = &m_local_address.get_socket_address();
    /**
     * In case there is one flow in the stream,
     * the stream will use @ref rmax_out_commit_chunk API call,
     * otherwise will use the @ref rmax_out_commit_chunk_to API call
     * in order to commit the data.
     */
    auto destination_address = &m_destination_flow.get_socket_address();
    if (destination_address) {
        flags |= RMAX_OUT_STREAM_REM_ADDR;
        m_rmax_parameters.remote_addr = const_cast<sockaddr*>(destination_address);
    }
    m_rmax_parameters.rate.rate_bps = m_rate.bps;
    m_rmax_parameters.rate.max_burst_in_pkt_num = m_rate.max_burst_in_packets;
    m_rmax_parameters.rate.typical_packet_sz = m_packet_app_header_size + m_packet_payload_size;

    if (m_rmax_parameters.rate.rate_bps) {
        flags |= RMAX_OUT_STREAM_RATE;
    }

    m_rmax_parameters.flags |= flags;
    m_rmax_parameters.max_chunk_size = m_num_of_packets_in_chunk;

    /**
     * This logic is required in order to implement the chunk management of the application.
     *
     * The application allocates more chunks than requested by the user, by the following logic.
     * Number of chunks will be the maximum between the amount of Rivermax internal TX HW buffer
     * chunks used plus one - '(OUT_STREAM_SIZE_DEFAULT / m_num_of_packets_in_chunk) + 1'
     * and the number of requested chunks by the user - m_num_of_requested_chunks.
     *
     * By that logic, the application will use the chunks container that is bigger than
     * actually owned by the HW as a cyclic buffer, by requesting to send the next
     * available chunk in a cyclic manner, ensuring that it will
     * always have more chunks than the HW capable to send.
     * Thus, when sending the next available buffer, the application will have one
     * that is not owned by the HW or will get a status of full TX HW buffer and will retry.
     */
    size_t num_of_rivermax_chunks = OUT_STREAM_SIZE_DEFAULT / m_num_of_packets_in_chunk;
    m_rmax_parameters.size_in_chunks = num_of_rivermax_chunks;
    m_num_of_chunks = std::max<size_t>(num_of_rivermax_chunks + 1, m_num_of_requested_chunks);
}

std::ostream& GenericSendStream::print(std::ostream& out) const
{
    ISendStream::print(out);

    out << "| Rate limit bps: " << m_rate.bps << "\n"
        << "| Rate limit max burst in packets: " << m_rate.max_burst_in_packets << "\n"
        << "| Memory length: " << m_mem_block.mem_block.length << "[B]" << "\n"
        << "| Number of user requested chunks: " << m_num_of_requested_chunks << "\n"
        << "| Number of application chunks: " << m_num_of_chunks << "\n"
        << "| Number of packets in chunk: " << m_num_of_packets_in_chunk << "\n"
        << "| Packet's payload size: " << m_packet_payload_size << "\n"
        << "+**********************************************\n";

    return out;
}

ReturnStatus GenericSendStream::get_next_chunk(GenericChunk* chunk)
{
    *chunk = *m_chunks[m_next_chunk_to_send_index];
    m_next_chunk_to_send_index++;
    m_next_chunk_to_send_index %= m_num_of_chunks;

    return ReturnStatus::success;
}

size_t GenericSendStream::initialize_chunks(void* pointer, rmax_mkey_id mkey)
{
    /* Initialize memory block */
    memset(&m_mem_block, 0, sizeof(m_mem_block));
    m_mem_block.mem_block.pointer = pointer;
    m_mem_block.mem_block.length = get_memory_length();
    m_mem_block.mkey_id = mkey;

    /* Initialize chunks */
    auto mem_offset = 0;
    for (size_t index = 0; index < m_num_of_chunks; index++) {
        rmax_chunk* chunk = new rmax_chunk;
        memset(chunk, 0, sizeof(*chunk));
        chunk->packets = new rmax_packet[m_num_of_packets_in_chunk];
        chunk->size = m_num_of_packets_in_chunk;
        /* Initialize packets */
        for (size_t pkt_indx = 0; pkt_indx < chunk->size; pkt_indx++) {
            auto& packet = chunk->packets[pkt_indx];
            memset(&packet, 0, sizeof(packet));
            packet.count = PACKET_IOVEC_SIZE;
            packet.iovec = new rmax_iov[packet.count];
            /* Initialize IO vector */
            for (size_t iovec_indx = 0; iovec_indx < packet.count; iovec_indx++) {
                auto& iovec = packet.iovec[iovec_indx];
                memset(&iovec, 0, sizeof(iovec));
                iovec.addr = reinterpret_cast<uint64_t>(pointer) + mem_offset;
                iovec.length = m_packet_app_header_size + m_packet_payload_size;
                iovec.mid = m_mem_block.mkey_id;
                mem_offset += iovec.length;
            }
        }
        m_chunks.push_back(new GenericChunk(chunk));
    }

    return m_mem_block.mem_block.length;
}

size_t GenericSendStream::get_memory_length() const
{
    return m_num_of_chunks
        * m_num_of_packets_in_chunk
        * PACKET_IOVEC_SIZE
        * (m_packet_app_header_size + m_packet_payload_size);
}

ReturnStatus GenericSendStream::create_stream()
{
    rmax_status_t status = rmax_out_create_gen_stream(&m_rmax_parameters, &m_stream_id);
    if (status != RMAX_OK) {
        std::cerr << "Failed to create generic stream with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

ReturnStatus GenericSendStream::destroy_stream()
{
    rmax_status_t status;

    do {
        status = rmax_out_destroy_stream(m_stream_id);
    } while (status == RMAX_ERR_BUSY);

    if (status != RMAX_OK) {
        std::cerr << "Failed to destroy generic stream with status: " << status << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus GenericSendStream::blocking_commit_chunk(
    GenericChunk& chunk, uint64_t timestamp_ns,
    rmax_commit_flags_t flags, TwoTupleFlow* dest_flow) const
{
    auto* _rmax_chunk = chunk.get_rmax_chunk();
    sockaddr* dest_address = nullptr;

    if (dest_flow != nullptr) {
        dest_address = &dest_flow->get_socket_address();
    }

    return blocking_commit_chunk_helper(_rmax_chunk, timestamp_ns, flags, dest_address);
}

inline ReturnStatus GenericSendStream::blocking_commit_chunk_helper(
    rmax_chunk* chunk, uint64_t timestamp_ns, rmax_commit_flags_t flags,
    sockaddr* flow, size_t retries) const
{
    ReturnStatus rc = ReturnStatus::success;
    rmax_status_t status;
    bool done = false;

    do {
        if (flow) {
            status = rmax_out_commit_chunk_to(m_stream_id, timestamp_ns, chunk, flags, flow);
        } else {
            status = rmax_out_commit_chunk(m_stream_id, timestamp_ns, chunk, flags);
        }

        if (likely(status == RMAX_OK)) {
            done = true;
        } else if (status == RMAX_SIGNAL) {
            done = true;
            rc = ReturnStatus::signal_received;
        } else if (unlikely(!retries-- || status == RMAX_ERR_HW_SEND_QUEUE_FULL)) {
            done = true;
            rc = ReturnStatus::hw_send_queue_full;
        } else {
            std::cerr << "Failed to commit with status: " << status << std::endl;
            rc = ReturnStatus::failure;
            done = true;
        }
    } while (likely(!done));

    return rc;
}
