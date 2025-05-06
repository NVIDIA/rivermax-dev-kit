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

#include <cstddef>
#include <iostream>
#include <ostream>
#include <cstring>

#include "rt_threads.h"

#include "rdk/io_node/common/rtp_video_send_stream.h"
#include "rdk/services/utils/defs.h"
#include "rdk/core/stream/send/media_stream.h"

using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::core;

RtpVideoSendStream::RtpVideoSendStream(const MediaStreamSettings& settings) :
    MediaSendStream(settings)
{
    memset(&m_send_stats, 0, sizeof(m_send_stats));
}

RtpVideoSendStream::RtpVideoSendStream(const MediaStreamSettings& settings, MediaStreamMemBlockset& mem_blocks) :
    MediaSendStream(settings, mem_blocks)
{
    memset(&m_send_stats, 0, sizeof(m_send_stats));
}

std::ostream& RtpVideoSendStream::print(std::ostream& out) const
{
    MediaSendStream::print(out);

    return out;
}

void RtpVideoSendStream::prepare_chunk_to_send(MediaChunk& chunk)
{
    auto chunk_length = chunk.get_length();
    byte_t* header_pointer;
    size_t header_stride_size;

    if (is_hds_on()) {
        header_pointer = reinterpret_cast<byte_t*>(chunk.get_app_hdr_ptr());
        header_stride_size = get_app_header_stride_size();
    } else {
        header_pointer = reinterpret_cast<byte_t*>(chunk.get_data_ptr());
        header_stride_size = get_data_stride_size();
    }

    uint64_t stride = 0;
    byte_t* current_packet_pointer;

    auto& media = m_stream_settings.m_media_settings;
    while (stride < chunk_length && m_send_stats.packet_counter < media.packets_in_frame_field) {
        current_packet_pointer = header_pointer + (stride * header_stride_size);
        build_2110_20_rtp_header(current_packet_pointer);
        if (!((stride + 1) % media.packets_in_line)) {
            // Prepare line number for next iteration:
            m_send_stats.line_number = (m_send_stats.line_number + 1) % media.resolution.height;
        }
        stride++;
    }
    m_send_stats.packet_counter %= media.packets_in_frame_field;
}

inline void RtpVideoSendStream::build_2110_20_rtp_header(byte_t* buffer)
{
    // build RTP header - 12 bytes:
    /*
     0                   1                   2                   3
     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     | V |P|X|  CC   |M|     PT      |            SEQ                |
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     |                           timestamp                           |
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     |                           ssrc                                |
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+*/
    buffer[0] = 0x80;  // 10000000 - version2, no padding, no extension.
    buffer[1] = 96;    // Payload type - Dynamic
    buffer[2] = (m_send_stats.rtp_sequence >> 8) & 0xff;  // Sequence number MSB.
    buffer[3] = (m_send_stats.rtp_sequence) & 0xff;  // Sequence number LSB.
    *(uint32_t*)&buffer[4] = htonl(m_send_stats.rtp_timestamp);
    *(uint32_t*)&buffer[8] = 0x0eb51dbd;  // Simulated SSRC.

    // build SRD header - 8-14 bytes:
    /* 0                   1                   2                   3
    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |    Extended Sequence Number   |           SRD Length          |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |F|     SRD Row Number          |C|         SRD Offset          |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ */
    buffer[12] = (m_send_stats.rtp_sequence >> 24) & 0xff;  // High 16 bits of Extended Sequence Number.
    buffer[13] = (m_send_stats.rtp_sequence >> 16) & 0xff;  // Low 16 bits of Extended Sequence Number.
    *(uint16_t*)&buffer[14] = htons(m_stream_settings.m_packet_payload_size - 20);  // SRD Length.

    auto& media = m_stream_settings.m_media_settings;
    uint16_t number_of_rows = media.resolution.height;
    if (media.video_scan_type == VideoScanType::Interlaced) {
        number_of_rows /= 2;
    }

    uint16_t srd_row_number = m_send_stats.line_number % number_of_rows;
    *(uint16_t*)&buffer[16] = htons(srd_row_number);
    buffer[16] |= (m_send_stats.rtp_interlace_field_indicator << 7);

    *(uint16_t*)&buffer[18] = htons(m_send_stats.srd_offset);  // SRD Offset.
    uint16_t group_size = (uint16_t)((m_stream_settings.m_packet_payload_size - 20) / 2.5);
    m_send_stats.srd_offset = (m_send_stats.srd_offset + group_size) %
            (group_size * media.packets_in_line);

    if (++m_send_stats.packet_counter == media.packets_in_frame_field) {
        buffer[1] |= 0x80; // Last packet in frame (Marker).
        // ST2210-20: the timestamp SHOULD be the same for each packet of the frame/field.
        auto fps_num = media.frame_rate.num;
        auto fps_denom = static_cast<double>(media.frame_rate.denom);
        double ticks = (media.sample_rate / (fps_num / fps_denom));
        if (media.video_scan_type == VideoScanType::Interlaced) {
            m_send_stats.rtp_interlace_field_indicator = !m_send_stats.rtp_interlace_field_indicator;
            ticks /= 2;
        }
        m_send_stats.rtp_timestamp += static_cast<uint32_t>(ticks);
    }
    m_send_stats.rtp_sequence++;
}

double RtpVideoSendStream::calculate_trs()
{
    double t_frame_ns;
    double r_active;
    uint32_t packets_in_frame;
    auto& media = m_stream_settings.m_media_settings;

    if (media.video_scan_type == VideoScanType::Progressive) {
        t_frame_ns = media.frame_field_time_interval_ns;
    }
    else {
        t_frame_ns = media.frame_field_time_interval_ns * 2;
    }

    if (media.video_scan_type == VideoScanType::Progressive) {
        r_active = (1080.0 / 1125.0);
    }
    else {
        if (media.resolution.height >= FHD_HEIGHT) { // As defined by SMPTE 2110-21 6.3.3
            r_active = (1080.0 / 1125.0);
        }
        else if (media.resolution.height >= 576) {
            r_active = (576.0 / 625.0);
        }
        else {
            r_active = (487.0 / 525.0);
        }
    }

    if (media.video_scan_type == VideoScanType::Progressive) {
        packets_in_frame = media.packets_in_frame_field;
    }
    else {
        packets_in_frame = media.packets_in_frame_field * 2;
    }

    return (t_frame_ns * r_active) / packets_in_frame;
}

double RtpVideoSendStream::calculate_send_time_ns(uint64_t time_now_ns)
{
    double send_time_ns = static_cast<double>(time_now_ns + NS_IN_SEC);
    double t_frame_ns;
    auto& media = m_stream_settings.m_media_settings;

    if (media.video_scan_type == VideoScanType::Progressive) {
        t_frame_ns = media.frame_field_time_interval_ns;
    }
    else {
        t_frame_ns = media.frame_field_time_interval_ns * 2;
    }

    uint64_t N = static_cast<uint64_t>(send_time_ns / t_frame_ns + 1);
    double first_packet_start_time_ns = N * t_frame_ns;  // Next alignment point.

    double r_active;
    double tro_default_multiplier;

    if (media.video_scan_type == VideoScanType::Progressive) {
        r_active = (1080.0 / 1125.0);
        if (media.resolution.height >= FHD_HEIGHT) {  // As defined by SMPTE 2110-21 6.3.2
            tro_default_multiplier = (43.0 / 1125.0);
        }
        else {
            tro_default_multiplier = (28.0 / 750.0);
        }
    }
    else {
        if (media.resolution.height >= FHD_HEIGHT) { // As defined by SMPTE 2110-21 6.3.3
            r_active = (1080.0 / 1125.0);
            tro_default_multiplier = (22.0 / 1125.0);
        }
        else if (media.resolution.height >= 576) {
            r_active = (576.0 / 625.0);
            tro_default_multiplier = (26.0 / 625.0);
        }
        else {
            r_active = (487.0 / 525.0);
            tro_default_multiplier = (20.0 / 525.0);
        }
    }

    uint32_t packets_in_frame;

    if (media.video_scan_type == VideoScanType::Progressive) {
        packets_in_frame = media.packets_in_frame_field;
    }
    else {
        packets_in_frame = media.packets_in_frame_field * 2;
    }

    double trs_ns = (t_frame_ns * r_active) / packets_in_frame;
    double tro = (tro_default_multiplier * t_frame_ns) - (VIDEO_TRO_DEFAULT_MODIFICATION * trs_ns);

    first_packet_start_time_ns += tro;

    m_send_stats.rtp_timestamp = static_cast<uint32_t>(
        time_to_rtp_timestamp(first_packet_start_time_ns,
                static_cast<int>(media.sample_rate)));
    send_time_ns = first_packet_start_time_ns;

    return send_time_ns;
}
