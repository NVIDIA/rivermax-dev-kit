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

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "rdk/core/stream/receive/ipo_receive_stream.h"
#include "rdk/services/utils/defs.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

IPOReceiveStream::IPOReceiveStream(const ipo_stream_settings_t& settings,
        const std::vector<IPOReceivePath>& paths, bool use_ext_seqn) :
    IAggregateReceiveStream(),
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

    ReturnStatus status = initialize_memory_layout();
    if (status != ReturnStatus::success) {
        throw std::runtime_error("Failed to initialize memory layout");
    }
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
    IAggregateReceiveStream::print(out)
        << "| Number of paths: " << m_paths.size() << "\n"
        << "| Maximum path differential: " << std::chrono::duration_cast<std::chrono::microseconds>(m_max_path_differential).count() << " us\n"
        << "\n";

    for(const auto& stream : m_streams) {
        stream.print(out);
    }

    out << "+**********************************************\n";

    return out;
}

ReturnStatus IPOReceiveStream::initialize_memory_layout()
{
    bool first = true;

    for (auto& stream : m_streams) {
        size_t header, payload;
        size_t header_stride, payload_stride;
        HeaderPayloadMemoryLayoutRequest sub_stream_memory_layout;
        ReturnStatus status = stream.determine_memory_layout(sub_stream_memory_layout);
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to determine memory layout for stream " << stream.get_id() << std::endl;
            return status;
        }
        std::tie(header, payload) = sub_stream_memory_layout.header_payload_buffers_size;
        header_stride = stream.get_header_stride_size();
        payload_stride = stream.get_payload_stride_size();
        if (first) {
            m_header_buffer_size = header;
            m_payload_buffer_size = payload;
            m_header_stride_size = header_stride;
            m_payload_stride_size = payload_stride;
            first = false;
        } else if (std::tie(header, payload, header_stride, payload_stride) !=
                   std::tie(m_header_buffer_size, m_payload_buffer_size, m_header_stride_size, m_payload_stride_size)) {
            std::cerr << "Redundant stream buffer sizes don't match" << std::endl;
            return ReturnStatus::failure;
        }
    }

    return ReturnStatus::success;
}

ReturnStatus IPOReceiveStream::determine_memory_layout(HeaderPayloadMemoryLayoutRequest& memory_layout_request) const
{

    if (!m_payload_buffer_size) {
        std::cerr << "Failed to determine memory layout, Payload size was not set" << std::endl;
        return ReturnStatus::failure;
    }

    memory_layout_request.header_payload_buffers_size = {m_header_buffer_size, m_payload_buffer_size};

    return ReturnStatus::success;
}

ReturnStatus IPOReceiveStream::validate_memory_layout(const HeaderPayloadMemoryLayoutResponse& memory_layout_response) const
{
    const auto& stream_memory_layout = memory_layout_response.memory_layout;

    if (stream_memory_layout.register_memory &&
        (stream_memory_layout.header_memory_keys.size() != m_streams.size() ||
        stream_memory_layout.payload_memory_keys.size() != m_streams.size())) {
        std::cerr << "Invalid number of memory keys" << std::endl;
        return ReturnStatus::failure;
    }

    if (!m_payload_buffer_size) {
        std::cerr << "Failed to validate memory layout, Payload size was not set" << std::endl;
        return ReturnStatus::failure;
    }

    if (stream_memory_layout.header_memory_size < m_header_buffer_size ||
        stream_memory_layout.payload_memory_size < m_payload_buffer_size) {
        std::cerr << "Invalid buffer size" << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus IPOReceiveStream::apply_memory_layout(const HeaderPayloadMemoryLayoutResponse& memory_layout_response)
{
    ReturnStatus status = validate_memory_layout(memory_layout_response);
    if (status != ReturnStatus::success) {
        std::cerr << "Failed to validate memory layout" << std::endl;
        return status;
    }
    const auto& stream_memory_layout = memory_layout_response.memory_layout;
    m_header_buffer = static_cast<byte_t*>(stream_memory_layout.header_memory_ptr);
    m_payload_buffer = static_cast<byte_t*>(stream_memory_layout.payload_memory_ptr);

    for (size_t i = 0; i < m_streams.size(); ++i) {
        HeaderPayloadMemoryLayoutResponse sub_stream_memory_layout_response;
        auto& sub_stream_memory_layout = sub_stream_memory_layout_response.memory_layout;
        sub_stream_memory_layout.header_memory_ptr = stream_memory_layout.header_memory_ptr;
        sub_stream_memory_layout.payload_memory_ptr = stream_memory_layout.payload_memory_ptr;
        sub_stream_memory_layout.header_memory_size = stream_memory_layout.header_memory_size;
        sub_stream_memory_layout.payload_memory_size = stream_memory_layout.payload_memory_size;
        sub_stream_memory_layout.register_memory = stream_memory_layout.register_memory;

        if (stream_memory_layout.register_memory) {
            sub_stream_memory_layout.header_memory_keys.push_back(stream_memory_layout.header_memory_keys[i]);
            sub_stream_memory_layout.payload_memory_keys.push_back(stream_memory_layout.payload_memory_keys[i]);
        }

        status = m_streams[i].apply_memory_layout(sub_stream_memory_layout_response);
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to set memory layout for stream " << m_streams[i].get_id() << std::endl;
            return status;
        }
    }

    return ReturnStatus::success;
}

ReturnStatus IPOReceiveStream::create_stream()
{
    m_sequence_number_wrap_around = get_sequence_number_wrap_around(m_num_of_packets_in_chunk);
    m_sequence_number_msb_mask = (get_sequence_number_mask() >> 1) + 1;
    // a half of the maximum offset where the difference changes sign
    m_sequence_number_init_offset = (get_sequence_number_mask() + 1ULL) / 4;

    ReturnStatus status;
    for (auto& stream : m_streams) {
        status = stream.create_stream();
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to create stream " << stream.get_id() << std::endl;
            return status;
        }
        m_chunks.emplace_back(stream.get_id(), m_header_data_split);
        stream.set_completion_moderation(0, m_settings.max_chunk_size, 0);
    }

    m_stream_created = true;

    return ReturnStatus::success;
}

ReturnStatus IPOReceiveStream::attach_flows()
{
    if (!m_stream_created) {
        std::cerr << "Failed to attach flows, the stream was not created" << std::endl;
        return ReturnStatus::failure;
    }

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

ReturnStatus IPOReceiveStream::detach_flows()
{
    if (!m_stream_created) {
        std::cerr << "Failed to dettach flows, the stream was not created" << std::endl;
        return ReturnStatus::failure;
    }

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
    if (!m_stream_created) {
        std::cerr << "Failed to destroy stream, the stream was not created" << std::endl;
        return ReturnStatus::failure;
    }

    bool success = true;

    for (auto& stream : m_streams) {
        ReturnStatus status = stream.destroy_stream();
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to destroy stream " << stream.get_id() << std::endl;
            success = false;
        }
    }

    m_stream_created = false;

    return (success) ? ReturnStatus::success : ReturnStatus::failure;
}

ReturnStatus IPOReceiveStream::sync_paths()
{
    if (!m_stream_created) {
        std::cerr << "Failed to sync paths, the stream was not created" << std::endl;
        return ReturnStatus::failure;
    }

    using namespace std::chrono;
    const auto end_time = steady_clock::now() + seconds(5);
    bool all_streams_is_ready = false;
    do {
        all_streams_is_ready = true;
        for (size_t i = 0; i < m_streams.size(); ++i) {
            auto& stream = m_streams.at(i);
            auto& chunk = m_chunks.at(i);
            ReturnStatus status = stream.get_next_chunk(chunk);
            if (status != ReturnStatus::success) {
                if (status != ReturnStatus::signal_received) {
                    std::cerr << "Failed to get data chunk from stream " << stream.get_id() << std::endl;
                }
                return status;
            }
            const auto comp = chunk.get_completion();
            if (rmx_input_get_completion_flag(comp, RMX_INPUT_COMPLETION_FLAG_MORE)) {
                all_streams_is_ready = false;
                break;
            }
        }
    } while (!all_streams_is_ready && end_time > steady_clock::now());
    if (!all_streams_is_ready) {
        std::cerr << "Failed to sync the streams!" << std::endl;
        std::cerr << "Neither of the streams should have RMX_INPUT_COMPLETION_FLAG_MORE indicated in their completion flags" << std::endl;
    }
    return ReturnStatus::success;
}

void IPOReceiveStream::start()
{
    m_state = State::WaitFirstPacket;
    m_start_time = clock::now() + m_max_path_differential;
}

ReturnStatus IPOReceiveStream::get_next_chunk(ReceiveChunk& chunk)
{
    /**
     * This function performs input stream reconstruction.
     *
     * On every call we check sub-streams for new packets. Having a completion
     * means that we already have packets placed into memory buffer by the
     * hardware.
     *
     * The initial state is @ref State::NotStarted. Calling @ref start method
     * or @ref get_next_chunk will mark current time as stream start time.
     *
     * The next state is @ref State::WaitFirstPacket. Here we're waiting for a
     * first input packet. Once the packet received, we're initializing the
     * internal buffer start's position (@ref m_index) to point to the first
     * packet. The next state is @ref State::Running.
     *
     * In @ref State::Running state, we're transferring a contiguous array of
     * received packets to the caller. Packets arrived during the first @ref
     * m_max_path_differential are skipped to make sure that the stream will be
     * completely reconstructed. A packet is transferred to the caller only
     * after residing in a buffer for at least @ref m_max_path_differential.
     * This should be enough for all late packets to arrive. @ref
     * ext_packet_info::is_valid is used to mark packets that aren't processed
     * yet. Once a packet transferred to the caller from any substream, this
     * flag is reset.
     *
     * If an array of received packets in a buffer is non-contiguous, we wait
     * for missed packet/packets arrival until the processing time of a first
     * valid packet that was placed to the buffer after current position
     * (arrived at @ref m_next_packet_time).
     *
     * If all the packets in a buffer were processed, we switch to @ref
     * State::Waiting state. The next arrived packet switches us back to @ref
     * State::Running. If the interval since the stream moved into
     * State::Waiting is more than @ref m_sender_restart_threshold, then the
     * first @ref m_max_path_differential are skipped as initial sync.
     */
    m_now = clock::now();
    if (unlikely(m_state == State::NotStarted)) {
        m_state = State::WaitFirstPacket;
        m_start_time = m_now + m_max_path_differential;
    }
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
    IPOReceiveChunk& ipo_chunk = static_cast<IPOReceiveChunk&>(chunk);
    ipo_chunk.set_completion_chunk_size(0);

    if (m_state != State::Running) {
        return ReturnStatus::success;
    }

    if (m_next_packet_time + m_max_path_differential >= m_now) {
        return ReturnStatus::success;
    }

    // search for the end of dropped packets interval
    auto start_idx = m_index;
    for (uint32_t iterations = 1; !m_ext_packet_info_arr[start_idx].is_valid; ++iterations) {
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

    // skip packets during initial sync
    while (true) {
        info = &m_ext_packet_info_arr[m_index];
        if (!info->is_valid) {
            // will skip not received packets on the next iteration
            return ReturnStatus::success;
        }
        if (info->timestamp >= m_start_time) {
            // sync is done
            break;
        }
        m_next_packet_time = info->timestamp;
        info->is_valid = false;
        ++m_index;
        if (m_index >= m_sequence_number_wrap_around) {
            m_index = 0;
        }
    }
    start_idx = m_index;

    size_t chunk_size = 0;
    // initialize completion content
    ipo_chunk.set_completion_seqn_first(info->sequence_number);
    ipo_chunk.set_completion_timestamp_first(info->hw_timestamp);

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
        ipo_chunk.set_completion_timestamp_last(info->hw_timestamp);
        complete_packet(info->sequence_number);

        ++m_index;
        chunk_size++;

        if (m_index >= m_sequence_number_wrap_around) {
            m_index = 0;
            // break the loop now otherwise return arrays will be non-contiguous
            break;
        }
    }

    ipo_chunk.set_completion_chunk_size(chunk_size);

    if (m_header_data_split) {
        ipo_chunk.set_completion_header_ptr(m_header_buffer + start_idx * m_header_stride_size);
    }
    ipo_chunk.set_completion_payload_ptr(m_payload_buffer + start_idx * m_payload_stride_size);
    if (m_pkt_info_enabled) {
        ipo_chunk.set_completion_info_ptr(&m_packet_info_arr[start_idx]);
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
        bool header_valid = get_sequence_number(ptr, packet_info.get_packet_sub_block_size(0), sequence_number);

        if (unlikely(!header_valid)) {
            handle_corrupted_packet(index, packet_info);
            continue;
        }

        uint32_t index_in_dest_arr = sequence_number % m_sequence_number_wrap_around;
        ext_packet_info& ext_info = m_ext_packet_info_arr[index_in_dest_arr];

        switch (m_state) {
        case State::NotStarted:
            assert(false && "invalid state");
            break;
        case State::Running:
            // do nothing
            break;
        case State::WaitFirstPacket:
            m_index = index_in_dest_arr;
            m_last_processed_sequence_number = (sequence_number - m_sequence_number_init_offset) & get_sequence_number_mask();
            m_state = State::Running;
            m_next_packet_time = m_now;
            break;
        case State::Waiting:
            if (m_now - m_next_packet_time > m_sender_restart_threshold) {
                // resetting input
                handle_sender_restart();
                m_index = index_in_dest_arr;
                m_last_processed_sequence_number = (sequence_number - m_sequence_number_init_offset) & get_sequence_number_mask();
                m_start_time = m_now + m_max_path_differential;
            }
            m_state = State::Running;
            m_next_packet_time = m_now;
            break;
        }
        uint32_t seq_delta = (sequence_number - m_last_processed_sequence_number) & get_sequence_number_mask();
        if (seq_delta == 0 || seq_delta & m_sequence_number_msb_mask) {
            // sequence number is less or equals to recently processed
            handle_late_packet(index, sequence_number, packet_info);
        } else if (!ext_info.is_valid || ext_info.sequence_number != sequence_number) {
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

void IPOReceiveStream::handle_late_packet(size_t index, uint32_t sequence_number, const ReceivePacketInfo& packet_info)
{
    NOT_IN_USE(index);
    NOT_IN_USE(sequence_number);
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
    m_last_processed_sequence_number = sequence_number;
}

void IPOReceiveStream::handle_sender_restart()
{
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
