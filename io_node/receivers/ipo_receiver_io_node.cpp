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

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <ostream>
#include <thread>
#include <vector>

#include <rivermax_api.h>

#include "rt_threads.h"

#include "receivers/ipo_receiver_io_node.h"
#include "api/rmax_apps_lib_api.h"

#define BREAK_ON_FAILURE(rc) if (unlikely(rc == ReturnStatus::failure)) { break; }

using namespace ral::io_node;
using namespace ral::lib::core;
using namespace ral::lib::services;

namespace {
static constexpr uint8_t RTP_HEADER_CSRC_GRANULARITY_BYTES = 4;
static constexpr uint32_t SEQUENCE_NUMBER_MASK_16BIT = 0xFFFF;
static constexpr uint32_t SEQUENCE_NUMBER_MASK_32BIT = 0xFFFFFFFF;
}

AppIPOReceiveStream::AppIPOReceiveStream(size_t id,
        const ipo_stream_settings_t& settings,
        bool extended_sequence_number,
        const std::vector<IPOReceivePath>& paths) :
    IPOReceiveStream(id, settings, paths,
            (extended_sequence_number) ? RMAX_IN_BUFFER_ATTER_STREAM_RTP_EXT_SEQN_PLACEMENT_ORDER : RMAX_IN_BUFFER_ATTER_STREAM_RTP_SEQN_PLACEMENT_ORDER),
    m_is_extended_sequence_number(extended_sequence_number),
    m_sequence_number_mask((extended_sequence_number) ? SEQUENCE_NUMBER_MASK_32BIT : SEQUENCE_NUMBER_MASK_16BIT)
{
    m_path_stats.resize(paths.size());
    m_path_packets.resize(settings.num_of_packets_in_chunk, std::vector<uint8_t>(paths.size(), 0));
}

ReturnStatus AppIPOReceiveStream::get_next_chunk(ReceiveChunk* chunk)
{
    ReturnStatus status = IPOReceiveStream::get_next_chunk(chunk);
    if (status == ReturnStatus::success) {
        m_statistic.rx_counter += chunk->get_length();

        for (uint32_t stride_index = 0; stride_index < chunk->get_length(); ++stride_index) {
            const auto& info = chunk->get_completion()->packet_info_arr[stride_index];
            m_statistic.received_bytes += info.hdr_size + info.data_size;
        }
    }
    return status;
}

void AppIPOReceiveStream::print_statistics(std::ostream& out, const std::chrono::high_resolution_clock::duration& duration) const
{
    std::stringstream ss;
    ss << "[stream_index " << std::setw(3) << get_id() << "]"
                << " Got " << std::setw(7) << m_statistic.rx_counter << " packets |"
                << " dropped: " << m_statistic.rx_dropped << " |"
                << " bad RTP hdr: " << m_statistic.rx_corrupt_rtp_header << " | ";
    ss << std::fixed << std::setprecision(2);
    double bitrate_Mbps = m_statistic.get_bitrate_Mbps();
    if (bitrate_Mbps > 1000.) {
        ss << std::setw(4) << bitrate_Mbps / 1000. << " Gbps during ";
    } else {
        ss << std::setw(4) << bitrate_Mbps << " Mbps during ";
    }
    ss << std::setw(4) << static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(duration).count()) / 1.e6 << " sec";

    for (uint32_t s_index = 0; s_index < m_paths.size(); ++s_index) {
        ss << " | " << m_paths[s_index].flow.get_destination_ip() << ":" << m_paths[s_index].flow.get_destination_port();
        if (m_statistic.rx_counter) {
            uint32_t number = static_cast<uint32_t>(ceil(100 * static_cast<double>(m_path_stats[s_index].rx_count) / static_cast<double>(m_statistic.rx_counter)));
            ss << " " << std::setw(3) << number << "%";
        } else {
            ss << "   0%";
        }
    }

    if (m_statistic.rx_dropped) {
        ss << std::endl << "ERROR !!! Drop Packets - count: " << m_statistic.rx_dropped;
    }
    if (m_statistic.rx_corrupt_rtp_header) {
        ss << std::endl << "ERROR !!! Corrupted Packets - count: " << m_statistic.rx_corrupt_rtp_header;
    }

    out << ss.str() << std::endl;
}

void AppIPOReceiveStream::reset_statistics()
{
    for (auto& stat : m_path_stats) {
        stat.reset();
    }
    m_statistic.reset();
}

void AppIPOReceiveStream::handle_corrupted_packet(size_t index, const rmax_in_packet_info& info)
{
    IPOReceiveStream::handle_corrupted_packet(index, info);

    ++m_statistic.rx_corrupt_rtp_header;
}

void AppIPOReceiveStream::handle_packet(size_t index, uint32_t sequence_number, const rmax_in_packet_info& info)
{
    IPOReceiveStream::handle_packet(index, sequence_number, info);

    auto& by_paths = m_path_packets.at(sequence_number % get_sequence_number_wrap_around());

    for (size_t i = 0; i < by_paths.size(); ++i) {
        by_paths[i] = (i == index) ? 1 : 0;
    }
}

void AppIPOReceiveStream::handle_redundant_packet(size_t index, uint32_t sequence_number, const rmax_in_packet_info& info)
{
    IPOReceiveStream::handle_redundant_packet(index, sequence_number, info);

    auto& by_paths = m_path_packets.at(sequence_number % get_sequence_number_wrap_around());

    by_paths[index] = 1;
}

void AppIPOReceiveStream::complete_packet(uint32_t sequence_number)
{
    IPOReceiveStream::complete_packet(sequence_number);

    auto& by_paths = m_path_packets.at(sequence_number % get_sequence_number_wrap_around());

    for (size_t i = 0; i < by_paths.size(); ++i) {
        m_path_stats[i].rx_count += by_paths[i];
    }

    // count dropped packets by sequence number
    if (m_initialized) {
        uint32_t expected = m_last_sequence_number + 1;
        uint32_t num_dropped = (sequence_number - expected) & get_sequence_number_mask();
        m_statistic.rx_dropped += num_dropped;
    }
    m_initialized = true;
    m_last_sequence_number = sequence_number;
}

bool AppIPOReceiveStream::get_sequence_number(const byte_ptr_t header, size_t length, uint32_t& sequence_number) const
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
        bool extended_sequence_number,
        const std::vector<std::string>& devices,
        size_t index, int cpu_core_affinity) :
    m_app_settings(app_settings),
    m_is_extended_sequence_number(extended_sequence_number),
    m_devices(devices),
    m_index(index),
    m_print_parameters(app_settings.print_parameters),
    m_cpu_core_affinity(cpu_core_affinity),
    m_sleep_between_operations(std::chrono::microseconds(app_settings.sleep_between_operations_us))
{
    m_stream_settings.stream_flags = RMAX_IN_CREATE_STREAM_INFO_PER_PACKET;
    m_stream_settings.packet_payload_size = m_app_settings.packet_payload_size;
    m_stream_settings.packet_app_header_size = m_app_settings.packet_app_header_size;
    m_stream_settings.num_of_packets_in_chunk = m_app_settings.num_of_packets_in_chunk;
    m_stream_settings.max_chunk_size = DEFAULT_MAX_CHUNK_SIZE;
    m_stream_settings.max_path_differential_us = max_path_differential_us;
}

std::ostream& IPOReceiverIONode::print(std::ostream& out) const
{
    out << "+#############################################\n"
        << "| Sender index: " << m_index << "\n"
        << "| Thread ID: 0x" << std::hex << std::this_thread::get_id() << std::dec << "\n"
        << "| CPU core affinity: " << m_cpu_core_affinity << "\n"
        << "| Number of streams in this thread: " << m_streams.size() << "\n"
        << "+#############################################\n";
    for (const auto& stream : m_streams) {
        stream->print(out);
    }
    return out;
}

void IPOReceiverIONode::initialize_streams(size_t start_id, const std::vector<std::vector<FourTupleFlow>>& flows)
{
    size_t id = start_id;
    m_streams.reserve(flows.size());
    for (const auto& flow_list : flows) {
        std::vector<IPOReceivePath> paths;

        paths.reserve(flow_list.size());
        assert(flow_list.size() == m_devices.size());
        for (size_t i = 0; i < flow_list.size(); ++i) {
            paths.emplace_back(m_devices[i], flow_list[i]);
        }
        m_streams.emplace_back(new AppIPOReceiveStream(id, m_stream_settings,
                    m_is_extended_sequence_number, std::move(paths)));
        ++id;
    }
}

void IPOReceiverIONode::print_parameters()
{
    if (!m_print_parameters) {
        return;
    }

    std::stringstream receiver_parameters;
    print(receiver_parameters);
    std::cout << receiver_parameters.str() << std::endl;
}

void IPOReceiverIONode::operator()()
{
    set_cpu_resources();
    ReturnStatus rc = create_streams();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to create receiver (" << m_index << ") streams" << std::endl;
        return;
    }
    rc = attach_flows();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to attach flows to receiver (" << m_index << ") streams" << std::endl;
        return;
    }
    print_parameters();

    ReceiveChunk chunk;
    rc = ReturnStatus::success;

    auto start_time = std::chrono::high_resolution_clock::now();
    while (likely(rc != ReturnStatus::failure && SignalHandler::get_received_signal() < 0)) {
        for (auto& stream : m_streams) {
            rc = stream->get_next_chunk(&chunk);
            BREAK_ON_FAILURE(rc);
        }

        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now - start_time;
        if (duration >= std::chrono::seconds{1}) {
            for (auto& stream : m_streams) {
                stream->print_statistics(std::cout, duration);
                stream->reset_statistics();
            }
            start_time = now;
        }
        if (m_sleep_between_operations.count() > 0) {
            std::this_thread::sleep_for(m_sleep_between_operations);
        }
    }

    rc = destroy_streams();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to destroy sender (" << m_index << ") streams" << std::endl;
        return;
    }
}

ReturnStatus IPOReceiverIONode::create_streams()
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

ReturnStatus IPOReceiverIONode::attach_flows()
{
    ReturnStatus rc;

    for (auto& stream : m_streams) {
        rc = stream->attach_flow();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to attach flow to stream (" << stream->get_id() << ")" << std::endl;
            return rc;
        }
    }

    return ReturnStatus::success;
}

ReturnStatus IPOReceiverIONode::detach_flows()
{
    ReturnStatus rc;

    for (auto& stream : m_streams) {
        rc = stream->detach_flow();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to detach flow from stream (" << stream->get_id() << ")" << std::endl;
            return rc;
        }
    }

    return ReturnStatus::success;
}

ReturnStatus IPOReceiverIONode::destroy_streams()
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

void IPOReceiverIONode::set_cpu_resources()
{
    memset(&m_cpu_affinity_mask, 0, sizeof(m_cpu_affinity_mask));
    if (m_cpu_core_affinity != CPU_NONE) {
        RMAX_CPU_SET(m_cpu_core_affinity, &m_cpu_affinity_mask);
    }
    rt_set_thread_affinity(&m_cpu_affinity_mask);
    rt_set_thread_priority(RMAX_THREAD_PRIORITY_TIME_CRITICAL - 1);
}
