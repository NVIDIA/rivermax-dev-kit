/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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

#include "receivers/rtp_receiver_io_node.h"
#include "api/rmax_apps_lib_api.h"
#include "services/utils/cpu.h"

using namespace ral::io_node;
using namespace ral::lib::core;
using namespace ral::lib::services;

namespace {
static constexpr uint8_t RTP_HEADER_CSRC_GRANULARITY_BYTES = 4;
static constexpr uint32_t SEQUENCE_NUMBER_MASK_16BIT = 0xFFFF;
static constexpr uint32_t SEQUENCE_NUMBER_MASK_32BIT = 0xFFFFFFFF;
}

AppRTPReceiveStream::AppRTPReceiveStream(size_t id, const ReceiveStreamSettings& settings,
            bool extended_sequence_number, bool header_data_split) :
    ReceiveStream(settings),
    m_stream_index(id),
    m_is_extended_sequence_number(extended_sequence_number),
    m_sequence_number_mask((extended_sequence_number) ? SEQUENCE_NUMBER_MASK_32BIT : SEQUENCE_NUMBER_MASK_16BIT),
    m_is_header_data_split(header_data_split)
{
}

ReturnStatus AppRTPReceiveStream::get_next_chunk(ReceiveChunk& chunk)
{
    ReturnStatus status = ReceiveStream::get_next_chunk(chunk);
    if (status != ReturnStatus::success) {
        return status;
    }
    m_statistic.rx_count += chunk.get_length();

    const byte_t* hdr_ptr;
    size_t stride_size;
    if (m_is_header_data_split) {
        hdr_ptr = reinterpret_cast<const byte_t*>(chunk.get_header_ptr());
        stride_size = get_header_stride_size();
    } else {
        hdr_ptr = reinterpret_cast<const byte_t*>(chunk.get_payload_ptr());
        stride_size = get_payload_stride_size();
    }
    for (uint32_t stride_index = 0; stride_index < chunk.get_length(); ++stride_index, hdr_ptr += stride_size) {
        auto info = chunk.get_packet_info(stride_index);
        size_t len = info.get_packet_sub_block_size(0);
        m_statistic.received_bytes += len;
        if (m_is_header_data_split) {
            len = info.get_packet_sub_block_size(1);
            m_statistic.received_bytes += len;
        }
        process_packet(hdr_ptr, len);
    }
    return status;
}

void AppRTPReceiveStream::print_statistics(std::ostream& out,
        const std::chrono::high_resolution_clock::duration& interval_duration) const
{
    using namespace std::chrono;

    std::stringstream ss;
    ss << "[stream_index " << std::setw(3) << m_stream_index << "]"
       << " Got " << std::setw(7) << m_statistic.rx_count << " packets |"
       << " dropped: " << m_statistic.rx_dropped
       << " |" << " bad RTP hdr: " << m_statistic.rx_corrupt_rtp_header
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

    if (m_statistic.rx_count) {
        uint32_t number = static_cast<uint32_t>(floor(100 * static_cast<double>(m_statistic.rx_count)
            / static_cast<double>(m_statistic.get_total_packets())));
        ss << " | " << std::setw(3) << number << "%";
    } else {
        ss << " |   0%";
    }

    if (m_statistic.rx_dropped) {
        ss << std::endl << "ERROR !!! Lost Packets - count: " << m_statistic.rx_dropped;
    }
    if (m_statistic.rx_corrupt_rtp_header) {
        ss << std::endl << "ERROR !!! Corrupted Packets - count: " << m_statistic.rx_corrupt_rtp_header;
    }

    out << ss.str() << std::endl;
}

void AppRTPReceiveStream::reset_statistics()
{
    m_statistic.reset();
}

void AppRTPReceiveStream::process_packet(const byte_t* header, size_t length)
{
    uint32_t sequence_number;
    bool is_valid_hdr = get_sequence_number(header, length, sequence_number);

    if (unlikely(!is_valid_hdr)) {
        ++m_statistic.rx_corrupt_rtp_header;
        return;
    }
    if (likely(m_initialized)) {
        uint32_t expected = m_last_sequence_number + 1;
        uint32_t num_dropped = (sequence_number - expected) & get_sequence_number_mask();
        m_statistic.rx_dropped += num_dropped;
    }

    m_initialized = true;
    m_last_sequence_number = sequence_number;
}

bool AppRTPReceiveStream::get_sequence_number(const byte_t* header, size_t length, uint32_t& sequence_number) const
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

RTPReceiverIONode::RTPReceiverIONode(
        const AppSettings& app_settings,
        bool extended_sequence_number,
        const std::string& device,
        size_t index, int cpu_core_affinity) :
    m_app_settings(app_settings),
    m_is_extended_sequence_number(extended_sequence_number),
    m_device(device),
    m_index(index),
    m_print_parameters(app_settings.print_parameters),
    m_cpu_core_affinity(cpu_core_affinity),
    m_sleep_between_operations(std::chrono::microseconds(app_settings.sleep_between_operations_us))
{
}

std::ostream& RTPReceiverIONode::print(std::ostream& out) const
{
    out << "+#############################################\n"
        << "| Receiver index: " << m_index << "\n"
        << "| Thread ID: 0x" << std::hex << std::this_thread::get_id() << std::dec << "\n"
        << "| CPU core affinity: " << m_cpu_core_affinity << "\n"
        << "| Number of streams in this thread: " << m_streams.size() << "\n"
        << "+#############################################\n";
    for (const auto& stream : m_streams) {
        stream->print(out);
    }
    return out;
}

void RTPReceiverIONode::initialize_streams(size_t start_id, const std::vector<FourTupleFlow>& flows)
{
    m_flows = flows;
    for (size_t id = start_id; id < start_id + m_flows.size(); ++id) {
        ReceiveStreamSettings stream_settings(TwoTupleFlow(id, m_device, 0),
                RMX_INPUT_APP_PROTOCOL_PACKET,
                RMX_INPUT_TIMESTAMP_RAW_NANO,
                {RMX_INPUT_STREAM_CREATE_INFO_PER_PACKET},
                m_app_settings.num_of_packets_in_chunk,
                m_app_settings.packet_payload_size,
                m_app_settings.packet_app_header_size);
        m_streams.emplace_back(new AppRTPReceiveStream(id, stream_settings,
                    m_is_extended_sequence_number,
                    m_app_settings.packet_app_header_size != 0));
    }
}

void RTPReceiverIONode::print_parameters()
{
    if (!m_print_parameters) {
        return;
    }

    std::stringstream receiver_parameters;
    print(receiver_parameters);
    std::cout << receiver_parameters.str() << std::endl;
}

ReturnStatus RTPReceiverIONode::wait_first_packet()
{
    ReturnStatus rc = ReturnStatus::success;
    bool initialized = false;
    while (likely(!initialized && rc != ReturnStatus::failure && SignalHandler::get_received_signal() < 0)) {
        for (auto& stream : m_streams) {
            ReceiveChunk chunk(stream->get_id(), m_app_settings.packet_app_header_size != 0);
            rc = stream->get_next_chunk(chunk);
            if (unlikely(rc != ReturnStatus::success)) {
                break;
            }
            if (chunk.get_length() > 0) {
                initialized = true;
                break;
            }
        }
    }
    return rc;
}

void RTPReceiverIONode::operator()()
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

    rc = wait_first_packet();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Error during waiting for a first packet" << std::endl;
    }
    // main receive loop
    auto start_time = std::chrono::high_resolution_clock::now();
    while (likely(rc != ReturnStatus::failure && SignalHandler::get_received_signal() < 0)) {
        for (auto& stream : m_streams) {
            ReceiveChunk chunk(stream->get_id(), m_app_settings.packet_app_header_size != 0);
            rc = stream->get_next_chunk(chunk);
            if (unlikely(rc == ReturnStatus::failure)) {
                std::cerr << "Error getting next chunk of packets" << std::endl;
                break;
            }
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

ReturnStatus RTPReceiverIONode::create_streams()
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

ReturnStatus RTPReceiverIONode::attach_flows()
{
    ReturnStatus rc;

    assert(m_flows.size() == m_streams.size());
    for (size_t i = 0; i < m_streams.size(); ++i) {
        auto& stream = m_streams[i];
        rc = stream->attach_flow(m_flows[i]);
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to attach flow to stream (" << stream->get_id() << ")" << std::endl;
            return rc;
        }
    }

    return ReturnStatus::success;
}

ReturnStatus RTPReceiverIONode::detach_flows()
{
    ReturnStatus rc;

    assert(m_flows.size() == m_streams.size());
    for (size_t i = 0; i < m_streams.size(); ++i) {
        auto& stream = m_streams[i];
        rc = stream->detach_flow(m_flows[i]);
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to detach flow from stream (" << stream->get_id() << ")" << std::endl;
            return rc;
        }
    }

    return ReturnStatus::success;
}

ReturnStatus RTPReceiverIONode::destroy_streams()
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

void RTPReceiverIONode::set_cpu_resources()
{
    set_current_thread_affinity(m_cpu_core_affinity);
    rt_set_thread_priority(RMAX_THREAD_PRIORITY_TIME_CRITICAL - 1);
}
