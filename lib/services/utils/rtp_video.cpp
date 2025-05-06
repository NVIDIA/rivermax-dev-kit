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

#include "rtp_video.h"
#include "services/utils/defs.h"

using namespace ral::lib::services;

static void compose_common_media_settings(AppSettings& s)
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

    s.media.sampling_type = SamplingType::Ycbcr422;
    s.media.bit_depth = 10;
    s.media.media_block_index = 0;
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
    s.num_of_chunks_in_mem_block = s.media.frames_fields_in_mem_block * s.media.chunks_in_frame_field;
    s.num_of_packets_in_mem_block = s.num_of_chunks_in_mem_block * s.num_of_packets_in_chunk;
}

static void compose_media_sdp(AppSettings& s) {
    std::string rate_string;
    if (s.media.frame_rate.denom == 1) {
        rate_string = std::to_string(s.media.frame_rate.num);
    } else {
        rate_string = std::to_string(s.media.frame_rate.num) + "/" +
                      std::to_string(s.media.frame_rate.denom);
    }

    std::string stream_type_suffix = s.media.stream_type == StreamType::VideoIPMX ? " IPMX" : "";

    std::stringstream sdp;
    sdp << "v=0\n"
        << "o=- 1443716955 1443716955 IN IP4 " << s.local_ip << "\n"
        << "s=SMPTE ST2110-20 narrow gap " << s.video_stream_type << "\n"
        << "t=0 0\n"
        << "m=video " << s.destination_port << " RTP/AVP 96\n"
        << "c=IN IP4 " << s.destination_ip << "/64\n"
        << "a=source-filter: incl IN IP4 " << s.destination_ip << " " << s.local_ip << "\n"
        << "a=rtpmap:96 raw/90000\n"
        << "a=fmtp:96 sampling=" << get_sampling_type_name(s.media.sampling_type)
        << "; width=" << s.media.resolution.width
        << "; height=" << s.media.resolution.height
        << "; exactframerate=" << rate_string << "; depth=" << s.media.bit_depth
        << "; TCS=SDR; colorimetry=BT709; PM=2110GPM; SSN=ST2110-20:2017; TP=2110TPN;"
        << stream_type_suffix << "\n"
        << "a=mediaclk:direct=0\n"
        << "a=ts-refclk:" << s.media.refclk;

    s.media.sdp = sdp.str();
}

void ral::lib::services::compose_ipmx_media_settings(AppSettings& s, const std::string& local_mac)
{
    compose_common_media_settings(s);

    s.media.stream_type = StreamType::VideoIPMX;
    if (s.ref_clk_is_ptp) {
        s.media.refclk = "ptp=IEEE1588-2008:39-A7-94-FF-FE-07-CB-D0:127";
    } else {
        s.media.refclk = "localmac=" + local_mac;
    }

    compose_media_sdp(s);
}

void ral::lib::services::compose_media_settings(AppSettings& s)
{
    compose_common_media_settings(s);

    s.media.stream_type = StreamType::Video2110_20;
    s.media.refclk = "ptp=IEEE1588-2008:39-A7-94-FF-FE-07-CB-D0:127";

    compose_media_sdp(s);
}

void ral::lib::services::calculate_tro_trs(media_settings_t& media_settings, double& tro, double& trs)
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

static const std::unordered_map<SamplingType, const char*> SUPPORTED_SAMPLING = {
    {SamplingType::Ycbcr422, "YCbCr-4:2:2"},
    {SamplingType::Ycbcr444, "YCbCr-4:4:4"},
    {SamplingType::Rgb,"RGB"}
};

const char* ral::lib::services::get_sampling_type_name(SamplingType sampling_type) {
    auto sampling = SUPPORTED_SAMPLING.find(sampling_type);
    if (sampling != SUPPORTED_SAMPLING.end()) {
        return sampling->second;
    } else {
        return "unknown";
    }
}
