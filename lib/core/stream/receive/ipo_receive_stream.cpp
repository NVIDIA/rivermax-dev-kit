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
#include <stdexcept>

#include "core/stream/receive/ipo_receive_stream.h"
#include "services/error_handling/error_handling.h"
#include "services/utils/defs.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

IPOReceiveStream::IPOReceiveStream(size_t id, const ipo_stream_settings_t& settings,
        const std::vector<IPOReceivePath>& paths, bool use_ext_seqn) :
    IAggregateStream(id),
    m_settings(settings),
    m_paths(paths),
    m_use_ext_seqn(use_ext_seqn),
    m_header_data_split(settings.packet_app_header_size > 0),
    m_pkt_info_enabled(settings.stream_options.count(RMX_INPUT_STREAM_CREATE_INFO_PER_PACKET) != 0),
    m_num_of_packets_in_chunk(static_cast<uint32_t>(settings.num_of_packets_in_chunk)),
    m_max_path_differential(std::chrono::microseconds(settings.max_path_differential_us))
{
    if (m_paths.size() == 0) {
        throw std::runtime_error("Must be at least one path");
    }

    m_ext_packet_info_arr.resize(settings.num_of_packets_in_chunk);
    m_packet_info_arr.resize(settings.num_of_packets_in_chunk);
    initialize_substreams();
}

void IPOReceiveStream::initialize_substreams()
{
    m_streams.reserve(m_paths.size());
    size_t id = 0;

    for (const auto& path : m_paths) {
        ReceiveStreamSettings settings(TwoTupleFlow(id, path.dev_ip, 0),
                RMX_INPUT_APP_PROTOCOL_PACKET,
                RMX_INPUT_TIMESTAMP_RAW_NANO,
                {RMX_INPUT_STREAM_CREATE_INFO_PER_PACKET},
                m_settings.num_of_packets_in_chunk,
                m_settings.packet_payload_size,
                m_settings.packet_app_header_size);

        if (m_use_ext_seqn) {
            settings.m_options.insert(RMX_INPUT_STREAM_RTP_EXT_SEQN_PLACEMENT_ORDER);
        } else {
            settings.m_options.insert(RMX_INPUT_STREAM_RTP_SEQN_PLACEMENT_ORDER);
        }
        m_streams.emplace_back(settings);
        ++id;
    }
}

std::ostream& IPOReceiveStream::print(std::ostream& out) const
{
    IAggregateStream::print(out)
        << "| Number of paths: " << m_paths.size() << "\n"
        << "| Maximum path differential: " << std::chrono::duration_cast<std::chrono::microseconds>(m_max_path_differential).count() << " us\n"
        << "\n";

    for(const auto& stream : m_streams) {
        stream.print(out);
    }

    out << "+**********************************************\n";

    return out;
}

ReturnStatus IPOReceiveStream::query_buffer_size(size_t& header_size, size_t& payload_size)
{
    bool first = true;

    for (auto& stream : m_streams) {
        size_t hdr, pld;
        size_t hdr_stride, pld_stride;

        ReturnStatus status = stream.query_buffer_size(hdr, pld);
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to query buffer size for stream " << stream.get_id() << std::endl;
            return status;
        }
        hdr_stride = stream.get_header_stride_size();
        pld_stride = stream.get_payload_stride_size();
        if (first) {
            header_size = hdr;
            payload_size = pld;
            m_header_stride_size = hdr_stride;
            m_payload_stride_size = pld_stride;
            first = false;
        } else if (std::tie(hdr, pld, hdr_stride, pld_stride) !=
                std::tie(header_size, payload_size, m_header_stride_size, m_payload_stride_size)) {
            std::cerr << "Redundant stream buffer sizes doesn't match" << std::endl;
            return ReturnStatus::failure;
        }
    }

    return ReturnStatus::success;
}

void IPOReceiveStream::set_buffers(void* header_ptr, void* payload_ptr)
{
    m_header_buffer = static_cast<byte_t*>(header_ptr);
    m_payload_buffer = static_cast<byte_t*>(payload_ptr);

    for (auto& stream : m_streams) {
        stream.set_buffers(header_ptr, payload_ptr);
    }
}

void IPOReceiveStream::set_memory_keys(const std::vector<rmx_mem_region>& header_regions,
        const std::vector<rmx_mem_region>& payload_regions)
{
    for (size_t i = 0; i < m_streams.size(); ++i) {
        m_streams[i].set_memory_keys(header_regions[i].mkey, payload_regions[i].mkey);
    }
}

ReturnStatus IPOReceiveStream::create_stream()
{
    m_sequence_number_wrap_around = get_sequence_number_wrap_around(m_num_of_packets_in_chunk);

    for (auto& stream : m_streams) {
        ReturnStatus status = stream.create_stream();
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to create stream " << stream.get_id() << std::endl;
            return status;
        }
        m_chunks.emplace_back(stream.get_id(), m_header_data_split);
        stream.set_completion_moderation(0, m_settings.max_chunk_size, 0);
    }

    return ReturnStatus::success;
}

ReturnStatus IPOReceiveStream::attach_flow()
{
    for (size_t i = 0; i < m_paths.size(); ++i) {
        auto& stream = m_streams.at(i);
        auto& path = m_paths.at(i);

        ReturnStatus status = stream.attach_flow(path.flow);
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to attach flow to stream " << stream.get_id() << std::endl;
            return status;
        }
    }

    return ReturnStatus::success;
}

ReturnStatus IPOReceiveStream::detach_flow()
{
    bool success = true;

    for (size_t i = 0; i < m_paths.size(); ++i) {
        auto& stream = m_streams.at(i);
        auto& path = m_paths.at(i);

        ReturnStatus status = stream.detach_flow(path.flow);
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to detach flow from stream " << stream.get_id() << std::endl;
            success = false;
        }
    }

    return (success) ? ReturnStatus::success : ReturnStatus::failure;
}

ReturnStatus IPOReceiveStream::destroy_stream()
{
    bool success = true;

    for (auto& stream : m_streams) {
        ReturnStatus status = stream.destroy_stream();
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to destroy stream " << stream.get_id() << std::endl;
            success = false;
        }
    }

    return (success) ? ReturnStatus::success : ReturnStatus::failure;
}

ReturnStatus IPOReceiveStream::get_next_chunk(IPOReceiveChunk* ipo_chunk)
{
    /**
     * This function performs input stream reconstruction.
     *
     * On every call we check sub-streams for new packets. Having a completion
     * means that we already have packets placed into memory buffer by the
     * hardware.
     *
     * The start state is @ref State::WaitFirstPacket. Here we're waiting for a
     * first input packet. Once the packet received, we're initializing the
     * internal buffer start's position (@ref m_index) to point to the first
     * packet. The next state is @ref State::Running.
     *
     * In @ref State::Running state, we're transferring a contiguous array of
     * received packets to the caller. A packet is transferred to the caller
     * only after residing in a buffer for at least
     * @ref m_max_path_differential. This should be enough for all late packets
     * to arrive. @ref ext_packet_info::is_valid is used to mark packets that
     * aren't processed yet. Once a packet transferred to the caller, this flag
     * is reset.
     *
     * If an array of received packets in a buffer is non-contiguous, we wait
     * for missed packet/packets arrival until the processing time of a first
     * valid packet that was placed to the buffer after current position
     * (arrived at @ref m_next_packet_time).
     *
     * If all the packets in a buffer were processed, we switch to @ref
     * State::Waiting state. The next arrived packet switches us back to @ref
     * State::Running.
     */
    if (!ipo_chunk) {
        std::cerr << "Chunk must be non-null!" << std::endl;
        return ReturnStatus::failure;
    }

    m_now = clock::now();
    for (size_t i = 0; i < m_streams.size(); ++i) {
        auto& stream = m_streams.at(i);
        ReturnStatus status = stream.get_next_chunk(m_chunks.at(i));
        if (status != ReturnStatus::success) {
            if (status != ReturnStatus::signal_received) {
                std::cerr << "Failed to get data chunk from stream " << stream.get_id() << std::endl;
            }
            return status;
        }
        process_completion(i, stream, m_chunks[i]);
    }

    ipo_chunk->set_completion_chunk_size(0);

    if (m_state != State::Running) {
        return ReturnStatus::success;
    }

    if (m_next_packet_time + m_max_path_differential >= m_now) {
        return ReturnStatus::success;
    }

    // search for the end of dropped packets interval
    auto start_idx = m_index;
    for (uint32_t iterations = 0; !m_ext_packet_info_arr[start_idx].is_valid; ++iterations) {
        ++start_idx;
        if (start_idx >= m_sequence_number_wrap_around) {
            start_idx = 0;
        }
        if (iterations == m_sequence_number_wrap_around) {
            m_state = State::Waiting;
            return ReturnStatus::success;
        }
    }

    // check that now is the time to process the current packet
    auto* info = &m_ext_packet_info_arr[start_idx];
    if (info->timestamp + m_max_path_differential >= m_now) {
        // the time to send the next chunk has not yet come
        m_next_packet_time = info->timestamp;
        return ReturnStatus::success;
    }
    // actually skipping the dropped packets because the packet at the end of
    // skipped interval must be processed now
    m_index = start_idx;

    size_t chunk_size = 0;
    // initialize completion content
    ipo_chunk->set_completion_seqn_first(info->sequence_number);
    ipo_chunk->set_completion_timestamp_first(info->hw_timestamp);

    // find maximum contiguous sequence of valid packets
    while (true) {
        info = &m_ext_packet_info_arr[m_index];
        if (!info->is_valid) {
            break;
        }
        // wait for packets from all paths to arrive for @ref m_max_path_differential time
        if (info->timestamp + m_max_path_differential >= m_now) {
            break;
        }

        info->is_valid = false;
        ipo_chunk->set_completion_timestamp_last(info->hw_timestamp);
        complete_packet(info->sequence_number);

        ++m_index;
        chunk_size++;

        if (m_index >= m_sequence_number_wrap_around) {
            m_index = 0;
            // break the loop now otherwise return arrays will be non-contiguous
            break;
        }
    }

    ipo_chunk->set_completion_chunk_size(chunk_size);

    if (m_header_data_split) {
        ipo_chunk->set_completion_header_ptr(m_header_buffer + start_idx * m_header_stride_size);
    }
    ipo_chunk->set_completion_payload_ptr(m_payload_buffer + start_idx * m_payload_stride_size);
    if (m_pkt_info_enabled) {
        ipo_chunk->set_completion_info_ptr(&m_packet_info_arr[start_idx]);
    }

    return ReturnStatus::success;
}

void IPOReceiveStream::process_completion(size_t index, const ReceiveStream& stream, ReceiveChunk& chunk)
{
    const byte_t* ptr;
    size_t stride_size;
    if (chunk.get_header_ptr()) {
        ptr = reinterpret_cast<const byte_t*>(chunk.get_header_ptr());
        stride_size = stream.get_header_stride_size();
    } else {
        ptr = reinterpret_cast<const byte_t*>(chunk.get_payload_ptr());
        stride_size = stream.get_payload_stride_size();
    }
    for (uint32_t stride_index = 0; stride_index < chunk.get_length(); ++stride_index, ptr += stride_size) {
        uint32_t sequence_number = 0;
        auto packet_info = chunk.get_packet_info(stride_index);
        bool hdr_valid = get_sequence_number(ptr, packet_info.get_packet_sub_block_size(0), sequence_number);

        if (unlikely(!hdr_valid)) {
            handle_corrupted_packet(index, packet_info);
            continue;
        }

        uint32_t index_in_dest_arr = sequence_number % m_sequence_number_wrap_around;
        ext_packet_info& ext_info = m_ext_packet_info_arr[index_in_dest_arr];

        switch (m_state) {
        case State::Running:
            // do nothing
            break;
        case State::WaitFirstPacket:
            m_index = index_in_dest_arr;
            /* fall through */
        case State::Waiting:
            m_state = State::Running;
            m_next_packet_time = m_now;
            break;
        }
        // If we haven't received yet packet with this sequeence number...
        if (!ext_info.is_valid) {
            // packet with this sequence number is received for the first time
            handle_packet(index, sequence_number, packet_info);
        } else {
            // packet with this sequence number was already received from the other path
            handle_redundant_packet(index, sequence_number, packet_info);
        }
    }
}

void IPOReceiveStream::handle_corrupted_packet(size_t index, const ReceivePacketInfo& packet_info)
{
    NOT_IN_USE(index);
    NOT_IN_USE(packet_info);
}

void IPOReceiveStream::handle_packet(size_t index, uint32_t sequence_number, const ReceivePacketInfo& packet_info)
{
    NOT_IN_USE(index);
    uint32_t index_in_dest_arr = sequence_number % m_sequence_number_wrap_around;
    ext_packet_info& ext_info = m_ext_packet_info_arr[index_in_dest_arr];

    if (m_pkt_info_enabled) {
        m_packet_info_arr[index_in_dest_arr] = packet_info;
    }

    ext_info.hw_timestamp = packet_info.get_packet_timestamp();
    ext_info.sequence_number = sequence_number;
    ext_info.is_valid = true;
    ext_info.timestamp = m_now;
}

void IPOReceiveStream::handle_redundant_packet(size_t index, uint32_t sequence_number, const ReceivePacketInfo& packet_info)
{
    NOT_IN_USE(index);
    NOT_IN_USE(sequence_number);
    NOT_IN_USE(packet_info);
}

void IPOReceiveStream::complete_packet(uint32_t sequence_number)
{
    NOT_IN_USE(sequence_number);
}

uint32_t IPOReceiveStream::get_sequence_number_wrap_around(uint32_t buffer_elements) const
{
    uint32_t seqn_wa = get_sequence_number_mask();
    if (seqn_wa >= buffer_elements) {
        seqn_wa = buffer_elements;
    } else {
        seqn_wa += 1;
    }

    return seqn_wa;
}
