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
#include <thread>
#include <utility>
#include <vector>

#include <rivermax_api.h>
#include "rt_threads.h"

#include "rdk/io_node/receivers/rtp_receiver_io_node.h"
#include "rdk/services/error_handling/error_handling.h"
#include "rdk/services/cpu/affinity/affinity.h"
#include "rdk/core/data_handler/receive_data_consumer_interface.h"

using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

AppRTPReceiveStream::AppRTPReceiveStream(const ReceiveStreamSettings& settings,
    bool is_extended_sequence_number, bool header_data_split, bool process_headers) :
    ReceiveStream(settings),
    m_is_extended_sequence_number(is_extended_sequence_number),
    m_sequence_number_mask(is_extended_sequence_number ? RTP_SEQUENCE_NUMBER_MASK_32BIT : RTP_SEQUENCE_NUMBER_MASK_16BIT),
    m_is_header_data_split(header_data_split),
    m_is_header_processing_enabled(process_headers)
{
}

void AppRTPReceiveStream::set_frame_start_handler(std::unique_ptr<IRTPEventHandler> event_handler)
{
    m_frame_notifier = std::move(event_handler);
}

ReturnStatus AppRTPReceiveStream::get_next_chunk(ReceiveChunk& chunk)
{
    ReturnStatus status = ReceiveStream::get_next_chunk(chunk);
    if (status != ReturnStatus::success) {
        return status;
    }
    m_statistic.rx_count += chunk.get_length();
    m_statistic_totals.rx_count += chunk.get_length();

    const byte_t* header_ptr;
    size_t stride_size;
    if (m_is_header_data_split) {
        header_ptr = reinterpret_cast<const byte_t*>(chunk.get_header_ptr());
        stride_size = get_header_stride_size();
    } else {
        header_ptr = reinterpret_cast<const byte_t*>(chunk.get_payload_ptr());
        stride_size = get_payload_stride_size();
    }
    for (uint32_t stride_index = 0; stride_index < chunk.get_length(); ++stride_index, header_ptr += stride_size) {
        auto info = chunk.get_packet_info(stride_index);
        size_t len = info.get_packet_sub_block_size(0);
        m_statistic.received_bytes += len;
        m_statistic_totals.received_bytes += len;
        if (m_is_header_data_split) {
            len = info.get_packet_sub_block_size(1);
            m_statistic.received_bytes += len;
            m_statistic_totals.received_bytes += len;
        }
        if (m_is_header_processing_enabled) {
            process_packet_header(header_ptr, len, info);
        }
    }
    return status;
}

void AppRTPReceiveStream::print_statistics(std::ostream& out,
    const std::chrono::high_resolution_clock::duration& interval_duration) const
{
    using namespace std::chrono;

    std::stringstream ss;
    ss << "[stream ID " << std::setw(3) << m_stream_id << "]"
       << " Got " << std::setw(7) << m_statistic.rx_count << " packets |"
       << " dropped: " << m_statistic.rx_dropped
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
    if (m_statistic.rx_corrupt_header) {
        ss << std::endl << "ERROR !!! Corrupted Packets - count: " << m_statistic.rx_corrupt_header;
    }

    out << ss.str() << std::endl;
}

void AppRTPReceiveStream::reset_statistics()
{
    m_statistic.reset();
}

void AppRTPReceiveStream::reset_statistics_totals()
{
    m_statistic_totals.reset();
}

void AppRTPReceiveStream::update_consumed_packets_stats(size_t consumed_packets, size_t unconsumed_packets)
{
    m_statistic.consumed_packets += consumed_packets;
    m_statistic.unconsumed_packets += unconsumed_packets;
    m_statistic_totals.consumed_packets += consumed_packets;
    m_statistic_totals.unconsumed_packets += unconsumed_packets;
}

RXStatistics AppRTPReceiveStream::get_statistics() const
{
    return m_statistic_totals;
}

void AppRTPReceiveStream::process_packet_header(const byte_t* header, size_t length,
    const ReceivePacketInfo& info)
{
    uint32_t sequence_number;
    bool is_valid_header = get_sequence_number(header, length, sequence_number);

    if (unlikely(!is_valid_header)) {
        ++m_statistic.rx_corrupt_header;
        ++m_statistic_totals.rx_corrupt_header;
        return;
    }
    if (likely(m_initialized)) {
        uint32_t expected = m_last_sequence_number + 1;
        uint32_t num_dropped = (sequence_number - expected) & get_sequence_number_mask();
        m_statistic.rx_dropped += num_dropped;
        m_statistic_totals.rx_dropped += num_dropped;
    }

    m_initialized = true;
    m_last_sequence_number = sequence_number;

    if (m_frame_notifier) {
        bool mbit = (header[1] & 0x80);
        if (!mbit && m_last_mbit) {
            uint32_t rtp_ts = static_cast<uint32_t>(header[7]) << 0 |
                              static_cast<uint32_t>(header[6]) << 8 |
                              static_cast<uint32_t>(header[5]) << 16 |
                              static_cast<uint32_t>(header[4]) << 24;
            uint64_t refclk_ts = info.get_packet_timestamp();
            m_frame_notifier->notify_rtp_event(rtp_ts, refclk_ts);
        }
        m_last_mbit = mbit;
    }
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
    bool is_extended_sequence_number,
    const std::vector<std::string>& devices,
    size_t index, int cpu_core_affinity,
    IONodeMemoryUtils& memory_utils) :
    ReceiverIONodeBase(app_settings, index, cpu_core_affinity, memory_utils),
    m_devices(devices),
    m_is_extended_sequence_number(is_extended_sequence_number)
{
}

void RTPReceiverIONode::initialize_streams(size_t start_id, const std::vector<ReceiveFlow>& flows)
{
    m_flows = flows;
    m_data_consumers.reserve(flows.size());
    for (size_t id = start_id; id < start_id + m_flows.size(); ++id) {
        ReceiveStreamSettings stream_settings(TwoTupleFlow(id, m_devices[0], 0),  // Currently supporting receiving on one device.
            RMX_INPUT_APP_PROTOCOL_PACKET,
            RMX_INPUT_TIMESTAMP_RAW_NANO,
            {RMX_INPUT_STREAM_CREATE_INFO_PER_PACKET},
            m_app_settings.num_of_packets_in_chunk,
            m_app_settings.packet_payload_size,
            m_app_settings.packet_app_header_size);
        m_streams.emplace_back(new AppRTPReceiveStream(stream_settings,
            m_is_extended_sequence_number,
            m_app_settings.packet_app_header_size != 0));
        m_data_consumers.emplace_back(new NullReceiveDataConsumer());
    }
}

void RTPReceiverIONode::assign_streams(size_t start_id, const std::vector<ReceiveFlow>& flows,
    std::vector<std::unique_ptr<IReceiveStream>>& streams)
{
    m_flows = flows;
    m_streams = std::move(streams);
    m_data_consumers.reserve(m_streams.size());
    for (auto& s : m_streams) {
        m_data_consumers.emplace_back(new NullReceiveDataConsumer());
    }
}

ReturnStatus RTPReceiverIONode::process_flows(bool is_attach)
{
    const char* operation_name = is_attach ? "attach" : "detach";
    const char* preposition = is_attach ? "to" : "from";

    if (m_flows.empty() || m_streams.empty()) {
        std::cerr << "No flows or streams to " << operation_name << std::endl;
        return ReturnStatus::failure;
    }

    if (m_flows.size() < m_streams.size()) {
        std::cerr << "Number of flows is less than number of streams" << std::endl;
        return ReturnStatus::failure;
    }

    size_t flows_per_stream = m_flows.size() / m_streams.size();
    size_t remainder_flows = m_flows.size() % m_streams.size();
    size_t flow_idx = 0;

    for (size_t stream_idx = 0; stream_idx < m_streams.size(); ++stream_idx) {
        auto stream = to_derived_stream(m_streams[stream_idx]);
        size_t stream_flow_count = flows_per_stream + (stream_idx < remainder_flows ? 1 : 0);

        for (size_t stream_flow_idx = 0; stream_flow_idx < stream_flow_count && flow_idx < m_flows.size();
             ++stream_flow_idx, ++flow_idx) {
            auto rc = is_attach ? stream->attach_flow(m_flows[flow_idx]) : stream->detach_flow(m_flows[flow_idx]);
            if (rc != ReturnStatus::success) {
                std::cerr << "Failed to " << operation_name << " flow " << flow_idx << " " << preposition
                    << " stream (" << stream->get_id() << ")" << std::endl;
                return rc;
            }
        }
    }

    return ReturnStatus::success;
}

ReturnStatus RTPReceiverIONode::consume_chunk(std::unique_ptr<IReceiveDataConsumer>& data_consumer,
    const ReceiveChunk& chunk, IReceiveStream& stream)
{
    size_t consumed_packets = 0;
    auto rc = ReceiverIONodeBase::consume_chunk(data_consumer, chunk, stream);
    if (rc != ReturnStatus::success) {
        return rc;
    }

    static_cast<AppRTPReceiveStream&>(stream).update_consumed_packets_stats(
        consumed_packets, chunk.get_length() - consumed_packets);
    return ReturnStatus::success;
}
