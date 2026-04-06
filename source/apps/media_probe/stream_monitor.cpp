/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <sstream>
#include <iomanip>
#include <cmath>

#include "rdk/services/media/media_defs.h"
#include "rdk/apps/media_probe/stream_monitor.h"
#include "rdk/apps/media_probe/media_probe.h"

using namespace std::chrono;
using namespace rdk::apps;

const OnNewFrameCallback null_new_frame_callback = [](const NewFrameEvent& event) {};


StreamMonitor::StreamMonitor(const MediaProbeSettings &app_settings, const ReceiveFlow& flow, size_t stream_index, MediaComponentId component_id) :
    m_app_settings(app_settings),
    m_flow(flow),
    m_packet_parser(false),
    m_stream_index(stream_index),
    m_component_id(component_id),
    m_on_new_frame_callback(null_new_frame_callback)
{
}

void StreamMonitor::measure_media_latency(uint64_t receive_ts, uint32_t rtp_ts)
{
    long double rtp_frequency = m_app_settings.media.sample_rate;
    uint32_t receive_ts_2_rtp_time;
    long double receive_ts_2_rtp_time_f;
    long double receive_ts_remainder;
    int32_t diff_in_rtp_time;
    long double media_latency;

    receive_ts_remainder = modfl(rtp_frequency * (receive_ts / duration_cast<duration<long double, std::nano>>(seconds{1}).count()),
        &receive_ts_2_rtp_time_f);
    receive_ts_2_rtp_time = static_cast<uint32_t>(receive_ts_2_rtp_time_f);
    /* calculate timestamp difference in RTP TS domain to handle 32-bit
     * wraparound */
    diff_in_rtp_time = static_cast<int32_t>(receive_ts_2_rtp_time - rtp_ts);
    media_latency = (diff_in_rtp_time + receive_ts_remainder) / rtp_frequency;
    m_media_latency_usec = media_latency * microseconds{ seconds{ 1 } }.count();
    if (unlikely(m_is_first_frame)) {
        m_media_latency_min_usec = m_media_latency_usec;
        m_media_latency_max_usec = m_media_latency_usec;
        return;
    }
    m_media_latency_min_usec = std::min(m_media_latency_min_usec, m_media_latency_usec);
    m_media_latency_max_usec = std::max(m_media_latency_max_usec, m_media_latency_usec);
}

void StreamMonitor::update_shared_stats()
{
    std::lock_guard<std::mutex> lock(m_shared_stats_mutex);
    m_shared_stats.received_packets_diff += m_received_packets - m_shared_stats.total_received_packets;
    m_shared_stats.total_received_packets = m_received_packets;
    m_shared_stats.missing_packets_diff += m_missing_packets - m_shared_stats.total_missing_packets;
    m_shared_stats.total_missing_packets = m_missing_packets;
    m_shared_stats.bad_rtp_headers_diff += m_bad_rtp_headers - m_shared_stats.total_bad_rtp_headers;
    m_shared_stats.total_bad_rtp_headers = m_bad_rtp_headers;
    m_shared_stats.received_frames_diff += m_received_frames - m_shared_stats.total_received_frames;
    m_shared_stats.total_received_frames = m_received_frames;
    m_shared_stats.packets_in_last_frame = m_packets_per_frame;
    m_shared_stats.fps = m_fps;
    m_shared_stats.media_latency_usec = m_media_latency_usec;
    m_shared_stats.media_latency_min_usec = m_media_latency_min_usec;
    m_shared_stats.media_latency_max_usec = m_media_latency_max_usec;
}

void StreamMonitor::process_new_frame(uint64_t receive_timestamp, uint32_t rtp_timestamp, uint32_t rtp_seq_num, const IReceiveStream& stream)
{
    m_received_frames++;
    uint32_t frame_packet_count = rtp_seq_num - m_prev_frame_seq_num;
    m_prev_frame_rtp_timestamp = rtp_timestamp;
    m_prev_frame_seq_num = rtp_seq_num;
    measure_media_latency(receive_timestamp, rtp_timestamp);

    NewFrameEvent event(m_component_id, stream, receive_timestamp, rtp_timestamp, rtp_seq_num, m_media_latency_usec);

    m_on_new_frame_callback(event);

    if (unlikely(m_is_first_frame)) {
        m_packets_per_frame = 0;
        m_is_first_frame = false;
    } else {
        m_packets_per_frame = frame_packet_count;
        m_fps = static_cast<float>(nanoseconds{ seconds{ 1 } }.count()) / (receive_timestamp - m_prev_frame_receive_timestamp);
    }
    m_prev_frame_receive_timestamp = receive_timestamp;
    update_shared_stats();
}

ReturnStatus StreamMonitor::consume_chunk(const ReceiveChunk& chunk, const IReceiveStream& stream, size_t& consumed_packets)
{
    const byte_t* packet_bodies = static_cast<const byte_t*>(chunk.get_payload_ptr());
    auto stride_size = stream.get_payload_stride_size();
    for (size_t pkt_idx = 0; pkt_idx < chunk.get_length(); pkt_idx++)
    {
        m_received_packets++;
        const ReceivePacketInfo& packet_info = chunk.get_packet_info(pkt_idx);
        size_t packet_size = packet_info.get_packet_sub_block_size(0);
        uint32_t rtp_seq_num;
        bool is_sequence_number_valid = m_packet_parser.get_sequence_number(packet_bodies, packet_size, true, rtp_seq_num);

        if (!is_sequence_number_valid) {
            m_bad_rtp_headers++;
            continue;
        }

        bool is_m_bit_set = ((m_packet_parser.rtp(packet_bodies)->mpt & RTP_M_BIT_MASK) != 0);
        uint64_t receive_timestamp = packet_info.get_packet_timestamp();
        uint32_t rtp_timestamp = htonl(m_packet_parser.rtp(packet_bodies)->timestamp);
        packet_bodies += stride_size;

        if (unlikely(m_is_first_packet)) {
            m_is_first_packet = false;
            m_prev_frame_rtp_timestamp = rtp_timestamp;
            m_prev_rtp_seq_num = rtp_seq_num;
            m_prev_frame_seq_num = rtp_seq_num;
            m_is_prev_mbit_set = is_m_bit_set;
            continue;
        }
        /*
         * Check if the sequence number is consecutive. Packet reordering or duplication
         * by the network is not allowed.
         */
        if (rtp_seq_num != m_prev_rtp_seq_num + 1) {
            m_missing_packets += rtp_seq_num - m_prev_rtp_seq_num - 1;
        } else if (m_is_prev_mbit_set) {
            /*
             * M bit is used in ST2110-20 and in ST2110-40. When adding support forST2110-30 audio,
             * M-bit should be ignored.
             */
             process_new_frame(receive_timestamp, rtp_timestamp, rtp_seq_num, stream);
        }
        m_prev_rtp_seq_num = rtp_seq_num;
        m_is_prev_mbit_set = is_m_bit_set;
    }
    consumed_packets = chunk.get_length();
    return ReturnStatus::success;
}

void StreamMonitor::print_and_reset_stats(std::ostream& out)
{
    std::stringstream oss;
    SharedStats stats;
    {
        std::lock_guard<std::mutex> lock(m_shared_stats_mutex);
        stats = m_shared_stats;
        m_shared_stats.received_packets_diff = 0;
        m_shared_stats.missing_packets_diff = 0;
        m_shared_stats.bad_rtp_headers_diff = 0;
        m_shared_stats.received_frames_diff = 0;
    }
    oss << "Flow: " << m_flow.get_destination_ip() << ":" << m_flow.get_destination_port()
        << ". Component: " << static_cast<size_t>(m_component_id)
        << ". Packets: " << stats.received_packets_diff
        << ", missing: " << stats.missing_packets_diff
        << " bad RTP: " << stats.bad_rtp_headers_diff
        << ". Frames: " << stats.received_frames_diff
        << " of size (pkts): " << stats.packets_in_last_frame
        << ". FPS: " << std::fixed << std::setprecision(2) << stats.fps
        << ". Media latency last: " << stats.media_latency_usec << " us"
        << ", min: " << stats.media_latency_min_usec << " us"
        << ", max: " << stats.media_latency_max_usec << " us"
        << std::endl;
    out << oss.str();
}
