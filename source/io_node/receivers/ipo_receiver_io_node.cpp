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

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <ostream>
#include <vector>

#include <rivermax_api.h>
#include "rt_threads.h"

#include "rdk/io_node/receivers/rtp_receiver_io_node.h"
#include "rdk/io_node/receivers/ipo_receiver_io_node.h"
#include "rdk/services/cpu/affinity/affinity.h"

#define BREAK_ON_FAILURE(rc) if (unlikely(rc == ReturnStatus::failure)) { break; }

using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

AppIPOReceiveStream::AppIPOReceiveStream(
        const ipo_stream_settings_t& settings,
        bool is_extended_sequence_number,
        const std::vector<IPOReceivePath>& paths) :
    IPOReceiveStream(settings, paths, is_extended_sequence_number),
    m_is_extended_sequence_number(is_extended_sequence_number),
    m_sequence_number_mask(is_extended_sequence_number ? RTP_SEQUENCE_NUMBER_MASK_32BIT : RTP_SEQUENCE_NUMBER_MASK_16BIT)
{
    m_statistic.path_stats.resize(paths.size());
    m_statistic_totals.path_stats.resize(paths.size());
    m_path_packets.resize(settings.num_of_packets_in_chunk, std::vector<uint8_t>(paths.size(), 0));
}

ReturnStatus AppIPOReceiveStream::get_next_chunk(ReceiveChunk& chunk)
{
    IPOReceiveChunk& ipo_chunk = static_cast<IPOReceiveChunk&>(chunk);

    ReturnStatus status = IPOReceiveStream::get_next_chunk(ipo_chunk);
    if (status == ReturnStatus::success) {
        m_statistic.rx_count += ipo_chunk.get_length();
        m_statistic_totals.rx_count += ipo_chunk.get_length();

        const auto packet_info_array = ipo_chunk.get_completion_info_ptr();
        for (uint32_t stride_index = 0; stride_index < ipo_chunk.get_length(); ++stride_index) {
            m_statistic.received_bytes += packet_info_array[stride_index].get_packet_sub_block_size(0);
            m_statistic_totals.received_bytes += packet_info_array[stride_index].get_packet_sub_block_size(0);
            if (m_header_data_split) {
                m_statistic.received_bytes += packet_info_array[stride_index].get_packet_sub_block_size(1);
                m_statistic_totals.received_bytes += packet_info_array[stride_index].get_packet_sub_block_size(1);
            }
        }
    }
    return status;
}

void AppIPOReceiveStream::print_statistics(std::ostream& out, const std::chrono::high_resolution_clock::duration& interval_duration) const
{
    using namespace std::chrono;

    std::stringstream ss;
    ss << "[stream_index " << std::setw(3) << get_id() << "]"
                << " Got " << std::setw(7) << m_statistic.rx_count << " packets |"
                << " dropped: ";
    for (uint32_t s_index = 0; s_index < m_paths.size(); ++s_index) {
        if (s_index > 0) {
            ss << ", ";
        }
        ss << m_statistic.path_stats[s_index].rx_dropped +  m_statistic.rx_dropped;
    }
    ss << " |" << " lost: " << m_statistic.rx_dropped
       << " |" << " exceed MD: " << m_statistic.rx_exceed_md
       << " |" << " bad RTP hdr: " << m_statistic.rx_corrupt_header
       << " | ";
    ss << std::fixed << std::setprecision(2);
    double duration_s = duration_cast<duration<double>>(interval_duration).count();
    double bitrate_Mbps = m_statistic.get_Mbits() / duration_s;
    if (bitrate_Mbps > 1000.) {
        ss << std::setw(4) << bitrate_Mbps / 1000. << " Gbps during ";
    } else {
        ss << std::setw(4) << bitrate_Mbps << " Mbps during ";
    }
    ss << std::setw(4) << duration_s << " sec";

    for (uint32_t s_index = 0; s_index < m_paths.size(); ++s_index) {
        ss << " | " << m_paths[s_index].flow.get_destination_ip() << ":" << m_paths[s_index].flow.get_destination_port();
        if (m_statistic.rx_count) {
            uint32_t number = static_cast<uint32_t>(floor(100 * static_cast<double>(
                m_statistic.path_stats[s_index].rx_count) / static_cast<double>(m_statistic.get_total_packets())));
            ss << " " << std::setw(3) << number << "%";
        } else {
            ss << "   0%";
        }
    }

    if (m_statistic.rx_dropped) {
        ss << std::endl << "ERROR !!! Lost Packets - count: " << m_statistic.rx_dropped;
    }
    if (m_statistic.rx_corrupt_header) {
        ss << std::endl << "ERROR !!! Corrupted Packets - count: " << m_statistic.rx_corrupt_header;
    }

    out << ss.str() << std::endl;
}

void AppIPOReceiveStream::reset_statistics()
{
    for (auto& stat : m_statistic.path_stats) {
        stat.reset();
    }
    m_statistic.reset();
}

void AppIPOReceiveStream::reset_statistics_totals()
{
    for (auto& stat : m_statistic_totals.path_stats) {
        stat.reset();
    }
    m_statistic_totals.reset();
}

void AppIPOReceiveStream::update_consumed_packets_stats(size_t consumed_packets, size_t unconsumed_packets)
{
    m_statistic.consumed_packets += consumed_packets;
    m_statistic.unconsumed_packets += unconsumed_packets;
    m_statistic_totals.consumed_packets += consumed_packets;
    m_statistic_totals.unconsumed_packets += unconsumed_packets;
}

IPORXStatistics AppIPOReceiveStream::get_statistics() const
{
    return m_statistic_totals;
}

void AppIPOReceiveStream::handle_corrupted_packet(size_t index, const ReceivePacketInfo& packet_info)
{
    IPOReceiveStream::handle_corrupted_packet(index, packet_info);

    ++m_statistic.rx_corrupt_header;
    ++m_statistic_totals.rx_corrupt_header;
}

void AppIPOReceiveStream::handle_late_packet(size_t index, uint32_t sequence_number, const ReceivePacketInfo& packet_info)
{
    IPOReceiveStream::handle_late_packet(index, sequence_number, packet_info);

    ++m_statistic.rx_exceed_md;
    ++m_statistic_totals.rx_exceed_md;
}

void AppIPOReceiveStream::handle_packet(size_t index, uint32_t sequence_number, const ReceivePacketInfo& packet_info)
{
    IPOReceiveStream::handle_packet(index, sequence_number, packet_info);

    auto& by_paths = m_path_packets.at(sequence_number % get_sequence_number_wrap_around());

    for (size_t i = 0; i < by_paths.size(); ++i) {
        by_paths[i] = (i == index) ? 1 : 0;
    }
}

void AppIPOReceiveStream::handle_redundant_packet(size_t index, uint32_t sequence_number, const ReceivePacketInfo& packet_info)
{
    IPOReceiveStream::handle_redundant_packet(index, sequence_number, packet_info);

    auto& by_paths = m_path_packets.at(sequence_number % get_sequence_number_wrap_around());

    by_paths[index] = 1;
}

void AppIPOReceiveStream::complete_packet(uint32_t sequence_number)
{
    IPOReceiveStream::complete_packet(sequence_number);

    auto& by_paths = m_path_packets.at(sequence_number % get_sequence_number_wrap_around());

    for (size_t i = 0; i < by_paths.size(); ++i) {
        m_statistic.path_stats[i].rx_count += by_paths[i];
        m_statistic.path_stats[i].rx_dropped += 1 - by_paths[i];
        m_statistic_totals.path_stats[i].rx_count += by_paths[i];
        m_statistic_totals.path_stats[i].rx_dropped += 1 - by_paths[i];
    }

    // count dropped packets by sequence number
    if (m_initialized) {
        uint32_t expected = m_last_sequence_number + 1;
        uint32_t num_dropped = (sequence_number - expected) & get_sequence_number_mask();
        m_statistic.rx_dropped += num_dropped;
        m_statistic_totals.rx_dropped += num_dropped;
    }
    m_initialized = true;
    m_last_sequence_number = sequence_number;
}

void AppIPOReceiveStream::handle_sender_restart()
{
    std::cout << "Sender restart detected" << std::endl;
    m_initialized = false;
}

bool AppIPOReceiveStream::get_sequence_number(const byte_t* header, size_t length, uint32_t& sequence_number) const
{
    if (length < 4 || (header[0] & 0xC0) != 0x80) {
        return false;
    }

    sequence_number = header[3] | header[2] << 8;
    if (m_is_extended_sequence_number) {
        uint8_t cc = 0x0F & header[0];
        uint8_t offset = cc * RTP_HEADER_CSRC_GRANULARITY_BYTES;
        sequence_number |= (header[offset + 12] << 24) | (header[offset + 13] << 16);
    }
    return true;
}

IPOReceiverIONode::IPOReceiverIONode(
    const AppSettings& app_settings,
    uint64_t max_path_differential_us,
    bool is_extended_sequence_number,
    const std::vector<std::string>& devices,
    size_t index, int cpu_core_affinity,
    IONodeMemoryUtils& memory_utils) :
    RTPReceiverIONode(app_settings, is_extended_sequence_number, devices, index, cpu_core_affinity,memory_utils)
{
    m_stream_settings.stream_options.insert(RMX_INPUT_STREAM_CREATE_INFO_PER_PACKET);
    m_stream_settings.packet_payload_size = m_app_settings.packet_payload_size;
    m_stream_settings.packet_app_header_size = m_app_settings.packet_app_header_size;
    m_stream_settings.num_of_packets_in_chunk = m_app_settings.num_of_packets_in_chunk;
    m_stream_settings.max_chunk_size = DEFAULT_MAX_CHUNK_SIZE;
    m_stream_settings.max_path_differential_us = max_path_differential_us;
}

void IPOReceiverIONode::initialize_streams(size_t start_id, const std::vector<std::vector<ReceiveFlow>>& flows)
{
    m_streams.reserve(flows.size());
    m_data_consumers.reserve(flows.size());
    for (const auto& flow_list : flows) {
        std::vector<IPOReceivePath> paths;

        paths.reserve(flow_list.size());
        assert(flow_list.size() == m_devices.size());
        for (size_t i = 0; i < flow_list.size(); ++i) {
            paths.emplace_back(m_devices[i], flow_list[i]);
        }
        m_streams.emplace_back(std::make_unique<AppIPOReceiveStream>(m_stream_settings,
            m_is_extended_sequence_number, std::move(paths)));
        m_data_consumers.emplace_back(std::make_unique<NullReceiveDataConsumer>());
    }
}

ReturnStatus IPOReceiverIONode::sync_streams()
{
    ReturnStatus rc = ReturnStatus::success;
    for (auto& stream : m_streams) {
        auto derived_stream = to_derived_stream(stream);
        rc = derived_stream->sync_paths();
        BREAK_ON_FAILURE(rc);
    }
    return rc;
}

std::unique_ptr<ReceiveChunk> IPOReceiverIONode::create_chunk(size_t stream_index) const
{
    return std::make_unique<IPOReceiveChunk>(m_stream_settings.packet_app_header_size != 0);
}

ReturnStatus IPOReceiverIONode::attach_flows()
{
    ReturnStatus rc;

    for (auto& stream : m_streams) {
        auto derived_stream = to_derived_stream(stream);
        rc = derived_stream->attach_flows();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to attach flow to stream (" << derived_stream->get_id() << ")" << std::endl;
            return rc;
        }
    }

    return ReturnStatus::success;
}

ReturnStatus IPOReceiverIONode::synchronous_start()
{
    ReturnStatus rc = sync_streams();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Error during initial sync" << std::endl;
        return rc;
    }
    for (auto& stream : m_streams) {
        auto derived_stream = to_derived_stream(stream);
        derived_stream->start();
    }
    rc = wait_first_packet();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Error during waiting for a first packet" << std::endl;
    }
    return rc;
}

ReturnStatus IPOReceiverIONode::detach_flows()
{
    ReturnStatus rc;

    for (auto& stream : m_streams) {
        auto derived_stream = to_derived_stream(stream);
        rc = derived_stream->detach_flows();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to detach flow from stream (" << derived_stream->get_id() << ")" << std::endl;
            return rc;
        }
    }

    return ReturnStatus::success;
}

ReturnStatus IPOReceiverIONode::consume_chunk(std::unique_ptr<IReceiveDataConsumer>& data_consumer,
    const ReceiveChunk& chunk, IReceiveStream& stream)
{
    size_t consumed_packets = 0;
    auto rc = ReceiverIONodeBase::consume_chunk(data_consumer, chunk, stream);
    if (rc != ReturnStatus::success) {
        return rc;
    }

    static_cast<AppIPOReceiveStream&>(stream).update_consumed_packets_stats(
        consumed_packets, chunk.get_length() - consumed_packets);
    return ReturnStatus::success;
}
