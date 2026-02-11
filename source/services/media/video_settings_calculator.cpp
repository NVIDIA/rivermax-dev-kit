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

#include <cstdint>
#include <string>
#include <unordered_map>

#include "rt_threads.h"

#include "rdk/services/media/video_settings_calculator.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/media/media_settings_video.h"
#include "rdk/services/sdp/sdp_defs.h"
#include "rdk/services/utils/defs.h"
#include "rdk/services/utils/enum_utils.h"

using namespace rivermax::dev_kit::services;

using BytesPerPixelRatio = std::pair<uint32_t, uint32_t>;
using VideoDepthPixelRatioMap =
    std::unordered_map<VideoSampling, std::unordered_map<VideoBitDepth, BytesPerPixelRatio>>;
/**
 * @brief: Map of video sampling formats to bytes-per-pixel ratios for each color bit depth.
 *
 * This map defines the pixel format characteristics by storing the ratio of bytes to pixels
 * for different video sampling types (RGB, YCbCr 4:4:4, YCbCr 4:2:2, YCbCr 4:2:0, KEY) and
 * video bit depths (8-bit, 10-bit, 12-bit). The ratio is expressed as a pair {bytes, pixels}
 * where the actual bytes-per-pixel value is calculated as bytes/pixels.
 *
 * The KEY sampling type is used for alpha channel in video.
 *
 * For example:
 * - RGB 8-bit: {3, 1} = 3 bytes per pixel (8 bits × 3 components ÷ 8 bits/byte)
 * - YCbCr 4:2:2 10-bit: {5, 2} = 2.5 bytes per pixel (10 bits × 2 pixels with shared chroma)
 * - KEY 10-bit: {4, 5} = 0.8 bytes per pixel (10 bits × 4 pixels)
 *
 * This is commonly referred to as the "pixel format stride" or "bytes per pixel" in video processing.
 */
const VideoDepthPixelRatioMap VIDEO_DEPTH_TO_PIXEL_RATIO = {
    {VideoSampling::RGB,
        {{VideoBitDepth::_8, {3, 1}},
         {VideoBitDepth::_10, {15, 4}},
         {VideoBitDepth::_12, {9, 2}}}},
    {VideoSampling::YCbCr_4_4_4,
        {{VideoBitDepth::_8, {3, 1}},
         {VideoBitDepth::_10, {15, 4}},
         {VideoBitDepth::_12, {9, 2}}}},
    {VideoSampling::YCbCr_4_2_2,
        {{VideoBitDepth::_8, {4, 2}},
         {VideoBitDepth::_10, {5, 2}},
         {VideoBitDepth::_12, {6, 2}}}},
    {VideoSampling::YCbCr_4_2_0,
        {{VideoBitDepth::_8, {6, 4}},
         {VideoBitDepth::_10, {15, 8}},
         {VideoBitDepth::_12, {9, 4}}}},
    {VideoSampling::KEY,
        {{VideoBitDepth::_8, {1, 1}},
         {VideoBitDepth::_10, {5, 4}},
         {VideoBitDepth::_12, {3, 2}}}}
};

bool ST_2110_20_MediaSettingsCalculator::is_bit_depth_supported(VideoSampling sampling, VideoBitDepth bit_depth)
{
    auto sampling_it = VIDEO_DEPTH_TO_PIXEL_RATIO.find(sampling);
    if (sampling_it != VIDEO_DEPTH_TO_PIXEL_RATIO.end()) {
        return sampling_it->second.find(bit_depth) != sampling_it->second.end();
    }
    return false;
}

void ST_2110_20_MediaSettingsCalculator::calculate_tro_trs(double& tro, double& trs) const
{
    double t_frame_ns;
    double r_active;
    double tro_default_multiplier;

    auto& video_settings = m_media_settings;

    if (video_settings.video_scan_type == VideoScanType::Progressive) {
        t_frame_ns = video_settings.media_unit_time_interval_ns;
    } else {
        t_frame_ns = video_settings.media_unit_time_interval_ns * 2;
    }

    if (video_settings.video_scan_type == VideoScanType::Progressive) {
        r_active = (1080.0 / 1125.0);
        if (video_settings.resolution.height >= _1080_HEIGHT) { // As defined by SMPTE 2110-21 6.3.2
            tro_default_multiplier = (43.0 / 1125.0);
        } else {
            tro_default_multiplier = (28.0 / 750.0);
        }
    } else {
        if (video_settings.resolution.height >= _1080_HEIGHT) { // As defined by SMPTE 2110-21 6.3.3
            r_active = (1080.0 / 1125.0);
            tro_default_multiplier = (22.0 / 1125.0);
        } else if (video_settings.resolution.height >= 576) {
            r_active = (576.0 / 625.0);
            tro_default_multiplier = (26.0 / 625.0);
        } else {
            r_active = (487.0 / 525.0);
            tro_default_multiplier = (20.0 / 525.0);
        }
    }

    uint32_t packets_in_frame;

    if (video_settings.video_scan_type == VideoScanType::Progressive) {
        packets_in_frame = video_settings.packets_in_media_unit;
    } else {
        packets_in_frame = video_settings.packets_in_media_unit * 2;
    }

    trs = (t_frame_ns * r_active) / packets_in_frame;
    tro = (tro_default_multiplier * t_frame_ns) - (VIDEO_TRO_DEFAULT_MODIFICATION * trs);
}

ReturnStatus ST_2110_20_MediaSettingsCalculator::calculate_media_settings()
{
    auto& video_settings = m_media_settings;

    if (!is_bit_depth_supported(video_settings.sampling_type, video_settings.bit_depth)) {
        std::cerr << "Unsupported width/sampling/bit depth combination: " << video_settings.resolution.width
            << " for sampling: " << enum_to_string(video_settings.sampling_type) << "\n";
        return ReturnStatus::failure;
    }

    if (video_settings.sampling_type == VideoSampling::YCbCr_4_2_0 &&
        video_settings.video_scan_type == VideoScanType::Interlaced) {
        std::cerr << "Error: " << enum_to_string(VideoSampling::YCbCr_4_2_0)
                  << " sampling is not permitted with interlaced video per SMPTE ST 2110-20\n";
        return ReturnStatus::failure;
    }

    auto bytes_per_pixel_ratio = VIDEO_DEPTH_TO_PIXEL_RATIO.at(video_settings.sampling_type).at(video_settings.bit_depth);
    uint32_t bytes_in_pgroup = bytes_per_pixel_ratio.first;
    uint32_t pixels_in_pgroup = bytes_per_pixel_ratio.second;
    uint32_t pgroups_in_line = (video_settings.resolution.width + pixels_in_pgroup - 1) / pixels_in_pgroup;

    video_settings.bytes_per_media_unit = bytes_in_pgroup * pgroups_in_line * video_settings.resolution.height;

    uint32_t pgroups_in_packet = 1; /* Non-zero initialization for Coverity to avoid a false "divide by zero" error */
    for (uint32_t pkt_cnt = 1; pkt_cnt <= pgroups_in_line; pkt_cnt++) {
        if (pgroups_in_line % pkt_cnt != 0) {
            continue;
        }
        /* We get here at least once, e.g. when pkt_cnt == pgroups_in_line, and pgroups_in_packet will be > 0 */
        pgroups_in_packet = pgroups_in_line / pkt_cnt;
        if (pgroups_in_packet * bytes_in_pgroup <= MediaSettings::MAX_PAYLOAD_SIZE) {
            break;
        }
    }

    video_settings.packets_in_line = pgroups_in_line / pgroups_in_packet;
    video_settings.packet_payload_size = static_cast<uint16_t>(pgroups_in_packet * bytes_in_pgroup + RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE);

    video_settings.protocol_header_size = RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE;
    video_settings.raw_packet_payload_size = static_cast<uint16_t>(pgroups_in_packet * bytes_in_pgroup);
    if (video_settings.header_data_split) {
        video_settings.packet_app_header_size = RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE;
        video_settings.packet_payload_size -= RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE;
    }

    video_settings.pixels_per_packet = static_cast<uint16_t>(pgroups_in_packet * pixels_in_pgroup);
    video_settings.packets_in_media_unit = static_cast<uint32_t>(video_settings.packets_in_line * video_settings.resolution.height);
    video_settings.media_unit_time_interval_ns = rational_cast<double>(NS_IN_SEC / video_settings.frame_rate);
    video_settings.lines_in_frame_field = video_settings.resolution.height;
    video_settings.ticks_per_media_unit = video_settings.sample_rate / video_settings.frame_rate;

    if (video_settings.video_scan_type == VideoScanType::Interlaced) {
        video_settings.packets_in_media_unit /= 2;
        video_settings.lines_in_frame_field /= 2;
        video_settings.media_unit_time_interval_ns /= 2;
        video_settings.ticks_per_media_unit /= 2;
        video_settings.bytes_per_media_unit /= 2;
    }

    bool chunk_size_applied = false;
    if (video_settings.packets_in_chunk) {
        if (video_settings.packets_in_media_unit % video_settings.packets_in_chunk == 0) {
            chunk_size_applied = true;
            std::cout << "Using custom chunk size: " << video_settings.packets_in_chunk << std::endl;
        } else {
            std::cout << "Custom chunk size (" << video_settings.packets_in_chunk
                << ") is ignored: must be divisor of packets in media unit ("
                << video_settings.packets_in_media_unit << ")" << std::endl;
        }
    }

    if (!chunk_size_applied) {
        constexpr size_t lines_in_chunk = 4;
        video_settings.packets_in_chunk = lines_in_chunk * video_settings.packets_in_line;
    }

    video_settings.chunks_in_media_unit =
        static_cast<size_t>(std::ceil(video_settings.packets_in_media_unit / static_cast<double>(video_settings.packets_in_chunk)));
    video_settings.chunks_in_mem_block = video_settings.media_units_in_mem_block * video_settings.chunks_in_media_unit;
    video_settings.packets_in_mem_block = video_settings.chunks_in_mem_block * video_settings.packets_in_chunk;

    video_settings.app_header_stride_size = align_up_pow2(video_settings.packet_app_header_size, get_cache_line_size());
    video_settings.data_stride_size = align_up_pow2(video_settings.packet_payload_size, get_cache_line_size());

    return ReturnStatus::success;
}

std::string ST_2110_20_MediaSettingsCalculator::generate_media_sdp(const std::vector<NetworkFlow>& flows)
{
    const auto& video_settings = m_media_settings;
    const auto& first_flow = flows[0];
    const bool is_multi_flow = flows.size() > 1;

    std::vector<std::string> group_mids;
    if (is_multi_flow) {
        for (size_t i = 0; i < flows.size(); ++i) {
            group_mids.push_back(std::string(1, 'a' + static_cast<char>(i)));
        }
    }

    SessionDescription::Builder session_builder(first_flow.source_ip);
    session_builder.set_session_id(SDPManager::generate_ntp_id())
        .set_session_version(SDPManager::generate_ntp_id() + 1)
        .set_session_name("SMPTE ST2110-20");
    if (is_multi_flow) {
        session_builder.add_group(GroupAttribute::Builder("DUP", group_mids).build());
    }

    auto time_description = TimeDescription::Builder().build();
    auto sdp_builder = SDPManager::Builder(session_builder.build(), std::move(time_description));

    for (size_t i = 0; i < flows.size(); ++i) {
        const auto& flow = flows[i];

        SMPTE2110_20_MediaDescription::Builder media_builder {
            flow.destination_port, TransportProtocol::RTP_AVP, std::to_string(video_settings.payload_type), flow.destination_ip};
        media_builder
            .set_source_filter(SourceFilterAttribute::Builder(flow.destination_ip, flow.source_ip).build())
            .set_smpte_standard_number(video_settings.smpte_standard_number)
            .set_sampling(video_settings.sampling_type)
            .set_width(video_settings.resolution.width)
            .set_height(video_settings.resolution.height)
            .set_exact_frame_rate(video_settings.frame_rate.to_total_string())
            .set_depth(video_settings.bit_depth)
            .set_video_scan_type(video_settings.video_scan_type)
            .set_colorimetry(video_settings.colorimetry)
            .set_timestamp_ref_clock(video_settings.ref_clk_is_ptp ? TimestampRefClock::PTP : TimestampRefClock::LocalMAC)
            .set_timestamp_ref_clock_ptp_traceable(video_settings.ref_clk_is_ptp && video_settings.refclk_id.empty())
            .set_timestamp_ref_clock_local_mac(video_settings.refclk_id)
            .set_extra_format_specific_parameters(m_extra_parameters);

        if (is_multi_flow) {
            media_builder.set_media_id(std::string(1, 'a' + static_cast<char>(i)));
        }

        sdp_builder.add_media_description(media_builder.build());
    }

    return sdp_builder.build()->to_string();
}

std::string ST_2110_20_MediaSettingsCalculator::get_smpte_standard_name() const
{
    const auto& video_settings = m_media_settings;
    if (video_settings.sampling_type == VideoSampling::KEY) {
        return "Video (Key)";
    }
    return "Video";
}

double ST_2110_20_MediaSettingsCalculator::get_transmit_offset_ns() const
{
    double tro, trs;
    calculate_tro_trs(tro, trs);
    return tro;
}

double ST_2110_20_MediaSettingsCalculator::align_time_to_media_unit_boundary_ns(uint64_t desired_time_ns) const
{
    double t_frame_ns;

    if (m_media_settings.video_scan_type == VideoScanType::Progressive) {
        t_frame_ns = m_media_settings.media_unit_time_interval_ns;
    }
    else {
        t_frame_ns = m_media_settings.media_unit_time_interval_ns * 2;
    }

    // Find the next aligned media unit start time
    uint64_t N = static_cast<uint64_t>(static_cast<double>(desired_time_ns) / t_frame_ns + 1);
    double first_packet_start_time_ns = N * t_frame_ns;

    return first_packet_start_time_ns;
}
