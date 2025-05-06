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

#include <thread>
#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>
#include <cstring>

#include <rivermax_api.h>

#include "rt_threads.h"

#include "senders/rtp_video_send_stream.h"
#include "api/rmax_apps_lib_api.h"

namespace ral
{
namespace io_node
{

constexpr uint16_t FHD_WIDTH = 1920;
constexpr uint16_t FHD_HEIGHT = 1080;
constexpr uint16_t UHD_HEIGHT = 2160;
constexpr uint16_t UHD_WIDTH = 3840;
constexpr uint8_t VIDEO_TRO_DEFAULT_MODIFICATION = 2;
constexpr size_t HD_PACKETS_PER_FRAME_422_10B = 4320;

void compose_media_settings(AppSettings& s)
{
    s.num_of_total_flows = s.num_of_total_streams;

    if (s.video_stream_type.compare(VIDEO_2110_20_1080p50) == 0) {
        s.media.resolution = { FHD_WIDTH, FHD_HEIGHT };
        s.media.frame_rate = { 50, 1 };
    } else if (s.video_stream_type.compare(VIDEO_2110_20_1080p60) == 0) {
        s.media.resolution = { FHD_WIDTH, FHD_HEIGHT };
        s.media.frame_rate = { 60, 1 };
    } else if (s.video_stream_type.compare(VIDEO_2110_20_2160p50) == 0) {
        s.media.resolution = { UHD_WIDTH, UHD_HEIGHT };
        s.media.frame_rate = { 50, 1 };
    } else if (s.video_stream_type.compare(VIDEO_2110_20_2160p60) == 0) {
        s.media.resolution = { UHD_WIDTH, UHD_HEIGHT };
        s.media.frame_rate = { 60, 1 };
    }

    std::string rate_string;
    if (s.media.frame_rate.denom == 1) {
        rate_string = std::to_string(s.media.frame_rate.num);
    } else {
        rate_string = std::to_string(s.media.frame_rate.num) + "/" +
                      std::to_string(s.media.frame_rate.denom);
    }

    std::stringstream sdp;
    sdp << "v=0\n"
        << "o=- 1443716955 1443716955 IN IP4 " << s.local_ip << "\n"
        << "s=SMPTE ST2110-20 narrow gap " << s.video_stream_type << "\n"
        << "t=0 0\n"
        << "m=video " << s.destination_port << " RTP/AVP 96\n"
        << "c=IN IP4 " << s.destination_ip << "/64\n"
        << "a=source-filter: incl IN IP4 " << s.destination_ip << " " << s.local_ip << "\n"
        << "a=rtpmap:96 raw/90000\n"
        << "a=fmtp:96 sampling=YCbCr-4:2:2; width=" << s.media.resolution.width
        << "; height=" << s.media.resolution.height
        << "; exactframerate=" << rate_string << "; depth=10;"
        << " TCS=SDR; colorimetry=BT709; PM=2110GPM; SSN=ST2110-20:2017; TP=2110TPN;\n"
        << "a=mediaclk:direct=0\n"
        << "a=ts-refclk:localmac=40-a3-6b-a0-2b-d2";

    s.media.sdp = sdp.str();
    s.media.media_block_index = 0;
    s.media.stream_type = StreamType::Video2110_20;
    s.media.video_scan_type = VideoScanType::Progressive;
    s.media.sample_rate = 90000;
    s.media.tp_mode = TPMode::TPN;
    s.media.packets_in_frame_field = HD_PACKETS_PER_FRAME_422_10B * \
        (s.media.resolution.width / FHD_WIDTH) * \
        (s.media.resolution.height / FHD_HEIGHT);

    s.num_of_memory_blocks = 1;
    s.packet_payload_size = 1220;  // Including RTP header.

    s.media.packets_in_line = s.media.packets_in_frame_field / s.media.resolution.height;

    bool chunk_size_applied = false;
    if (s.num_of_packets_in_chunk_specified) {
        if (s.media.packets_in_frame_field % s.num_of_packets_in_chunk == 0) {
            chunk_size_applied = true;
            if (s.print_parameters) {
                std::cout << "Using custom chunk size: " << s.num_of_packets_in_chunk << std::endl;
            }
        } else {
            std::cout << "Custom chunk size is ignored: must be divisor of packets in field ("
                      << s.media.packets_in_frame_field << ")" << std::endl;
        }
    }
    if (!chunk_size_applied) {
        const size_t lines_in_chunk = 4;
        s.num_of_packets_in_chunk = lines_in_chunk * s.media.packets_in_line;
    }

    s.media.frame_field_time_interval_ns = NS_IN_SEC / static_cast<double>(
        s.media.frame_rate.num) / s.media.frame_rate.denom;
    s.media.lines_in_frame_field = s.media.resolution.height;

    if (s.media.video_scan_type == VideoScanType::Interlaced) {
        s.media.packets_in_frame_field /= 2;
        s.media.lines_in_frame_field /= 2;
        s.media.frame_field_time_interval_ns /= 2;
    }

    s.media.chunks_in_frame_field = static_cast<size_t>(std::ceil(
        s.media.packets_in_frame_field / static_cast<double>(s.num_of_packets_in_chunk)));
    s.media.frames_fields_in_mem_block = 1;
    s.num_of_chunks_in_mem_block = s.media.frames_fields_in_mem_block * s.media.chunks_in_frame_field;
    s.num_of_packets_in_mem_block = s.num_of_chunks_in_mem_block * s.num_of_packets_in_chunk;
}

void calculate_tro_trs(media_settings_t& media_settings, double& tro, double& trs)
{
    double t_frame_ns;
    double r_active;
    double tro_default_multiplier;

    if (media_settings.video_scan_type == VideoScanType::Progressive) {
        t_frame_ns = media_settings.frame_field_time_interval_ns;
    }
    else {
        t_frame_ns = media_settings.frame_field_time_interval_ns * 2;
    }

    if (media_settings.video_scan_type == VideoScanType::Progressive) {
        r_active = (1080.0 / 1125.0);
        if (media_settings.resolution.height >= FHD_HEIGHT) {  // As defined by SMPTE 2110-21 6.3.2
            tro_default_multiplier = (43.0 / 1125.0);
        }
        else {
            tro_default_multiplier = (28.0 / 750.0);
        }
    }
    else {
        if (media_settings.resolution.height >= FHD_HEIGHT) { // As defined by SMPTE 2110-21 6.3.3
            r_active = (1080.0 / 1125.0);
            tro_default_multiplier = (22.0 / 1125.0);
        }
        else if (media_settings.resolution.height >= 576) {
            r_active = (576.0 / 625.0);
            tro_default_multiplier = (26.0 / 625.0);
        }
        else {
            r_active = (487.0 / 525.0);
            tro_default_multiplier = (20.0 / 525.0);
        }
    }

    uint32_t packets_in_frame;

    if (media_settings.video_scan_type == VideoScanType::Progressive) {
        packets_in_frame = media_settings.packets_in_frame_field;
    }
    else {
        packets_in_frame = media_settings.packets_in_frame_field * 2;
    }

    trs = (t_frame_ns * r_active) / packets_in_frame;
    tro = (tro_default_multiplier * t_frame_ns) - (VIDEO_TRO_DEFAULT_MODIFICATION * trs);
}

}
}

using namespace ral::io_node;
using namespace ral::lib::core;
using namespace ral::lib::services;

RTPVideoSendStream::RTPVideoSendStream(const MediaStreamSettings& settings, MediaStreamMemBlockset& mem_blocks) :
    MediaSendStream(settings, mem_blocks),
    m_media_settings(settings.m_media_settings)
{
    memset(&m_send_stats, 0, sizeof(m_send_stats));
}

std::ostream& RTPVideoSendStream::print(std::ostream& out) const
{
    MediaSendStream::print(out);

    return out;
}

void RTPVideoSendStream::prepare_chunk_to_send(MediaChunk& chunk)
{
    byte_t* data_pointer = reinterpret_cast<byte_t*>(chunk.get_data_ptr());
    auto chunk_length = chunk.get_length();
    // TODO: Update this when adding header data split.
    auto data_stride_size = get_data_stride_size();
    uint64_t stride = 0;
    byte_t* current_packet_pointer;

    while (stride < chunk_length && m_send_stats.packet_counter < m_media_settings.packets_in_frame_field) {
        current_packet_pointer = data_pointer + (stride * data_stride_size);
        build_2110_20_rtp_header(current_packet_pointer);
        if (!((stride + 1) % m_media_settings.packets_in_line)) {
            // Prepare line number for next iteration:
            m_send_stats.line_number = (m_send_stats.line_number + 1) % m_media_settings.resolution.height;
        }
        stride++;
    }
    m_send_stats.packet_counter %= m_media_settings.packets_in_frame_field;
}

inline void RTPVideoSendStream::build_2110_20_rtp_header(byte_t* buffer)
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
    buffer[1] = 0;  // Payload type.
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

    uint16_t number_of_rows = m_media_settings.resolution.height;
    if (m_media_settings.video_scan_type == VideoScanType::Interlaced) {
        number_of_rows /= 2;
    }

    uint16_t srd_row_number = m_send_stats.line_number % number_of_rows;
    *(uint16_t*)&buffer[16] = htons(srd_row_number);
    buffer[16] |= (m_send_stats.rtp_interlace_field_indicator << 7);

    *(uint16_t*)&buffer[18] = htons(m_send_stats.srd_offset);  // SRD Offset.
    uint16_t group_size = (uint16_t)((m_stream_settings.m_packet_payload_size - 20) / 2.5);
    m_send_stats.srd_offset = (m_send_stats.srd_offset + group_size) %
            (group_size * m_media_settings.packets_in_line);

    if (++m_send_stats.packet_counter == m_media_settings.packets_in_frame_field) {
        buffer[1] |= 0x80; // Last packet in frame (Marker).
        // ST2210-20: the timestamp SHOULD be the same for each packet of the frame/field.
        auto fps_num = m_media_settings.frame_rate.num;
        auto fps_denom = static_cast<double>(m_media_settings.frame_rate.denom);
        double ticks = (m_media_settings.sample_rate / (fps_num / fps_denom));
        if (m_media_settings.video_scan_type == VideoScanType::Interlaced) {
            m_send_stats.rtp_interlace_field_indicator = !m_send_stats.rtp_interlace_field_indicator;
            ticks /= 2;
        }
        m_send_stats.rtp_timestamp += static_cast<uint32_t>(ticks);
    }
    m_send_stats.rtp_sequence++;
}

double RTPVideoSendStream::calculate_trs()
{
    double t_frame_ns;
    double r_active;
    uint32_t packets_in_frame;

    if (m_media_settings.video_scan_type == VideoScanType::Progressive) {
        t_frame_ns = m_media_settings.frame_field_time_interval_ns;
    }
    else {
        t_frame_ns = m_media_settings.frame_field_time_interval_ns * 2;
    }

    if (m_media_settings.video_scan_type == VideoScanType::Progressive) {
        r_active = (1080.0 / 1125.0);
    }
    else {
        if (m_media_settings.resolution.height >= FHD_HEIGHT) { // As defined by SMPTE 2110-21 6.3.3
            r_active = (1080.0 / 1125.0);
        }
        else if (m_media_settings.resolution.height >= 576) {
            r_active = (576.0 / 625.0);
        }
        else {
            r_active = (487.0 / 525.0);
        }
    }

    if (m_media_settings.video_scan_type == VideoScanType::Progressive) {
        packets_in_frame = m_media_settings.packets_in_frame_field;
    }
    else {
        packets_in_frame = m_media_settings.packets_in_frame_field * 2;
    }

    return (t_frame_ns * r_active) / packets_in_frame;
}

double RTPVideoSendStream::calculate_send_time_ns(uint64_t time_now_ns)
{
    double send_time_ns = static_cast<double>(time_now_ns + NS_IN_SEC);
    double t_frame_ns;

    if (m_media_settings.video_scan_type == VideoScanType::Progressive) {
        t_frame_ns = m_media_settings.frame_field_time_interval_ns;
    }
    else {
        t_frame_ns = m_media_settings.frame_field_time_interval_ns * 2;
    }

    uint64_t N = static_cast<uint64_t>(send_time_ns / t_frame_ns + 1);
    double first_packet_start_time_ns = N * t_frame_ns;  // Next alignment point.

    double r_active;
    double tro_default_multiplier;

    if (m_media_settings.video_scan_type == VideoScanType::Progressive) {
        r_active = (1080.0 / 1125.0);
        if (m_media_settings.resolution.height >= FHD_HEIGHT) {  // As defined by SMPTE 2110-21 6.3.2
            tro_default_multiplier = (43.0 / 1125.0);
        }
        else {
            tro_default_multiplier = (28.0 / 750.0);
        }
    }
    else {
        if (m_media_settings.resolution.height >= FHD_HEIGHT) { // As defined by SMPTE 2110-21 6.3.3
            r_active = (1080.0 / 1125.0);
            tro_default_multiplier = (22.0 / 1125.0);
        }
        else if (m_media_settings.resolution.height >= 576) {
            r_active = (576.0 / 625.0);
            tro_default_multiplier = (26.0 / 625.0);
        }
        else {
            r_active = (487.0 / 525.0);
            tro_default_multiplier = (20.0 / 525.0);
        }
    }

    uint32_t packets_in_frame;

    if (m_media_settings.video_scan_type == VideoScanType::Progressive) {
        packets_in_frame = m_media_settings.packets_in_frame_field;
    }
    else {
        packets_in_frame = m_media_settings.packets_in_frame_field * 2;
    }

    double trs_ns = (t_frame_ns * r_active) / packets_in_frame;
    double tro = (tro_default_multiplier * t_frame_ns) - (VIDEO_TRO_DEFAULT_MODIFICATION * trs_ns);

    first_packet_start_time_ns += tro;

    m_send_stats.rtp_timestamp = static_cast<uint32_t>(
        time_to_rtp_timestamp(first_packet_start_time_ns,
                static_cast<int>(m_media_settings.sample_rate)));
    send_time_ns = first_packet_start_time_ns;

    return send_time_ns;
}
