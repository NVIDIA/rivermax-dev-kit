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

#include <algorithm>
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

using namespace rdk::services;

/**
 * @brief: Number of bytes, pixel columns, and pixel rows in one SMPTE ST 2110-20 pgroup.
 */
struct PgroupGeometry
{
    uint32_t octets;
    uint32_t columns;
    uint32_t rows;
};

using VideoDepthPgroupGeometryMap =
    std::unordered_map<VideoSampling, std::unordered_map<VideoBitDepth, PgroupGeometry>>;
/**
 * @brief: Map of video sampling formats to pgroup geometry for each color bit depth.
 *
 * Each entry is {octets, columns, rows}. Most pgroups cover one raster row.
 * YCbCr 4:2:0 pgroups cover two consecutive rows, so their horizontal
 * coverage must not be flattened into the total number of covered samples.
 *
 * The KEY sampling type is used for the alpha channel.
 */
const VideoDepthPgroupGeometryMap VIDEO_DEPTH_TO_PGROUP_GEOMETRY = {
    {VideoSampling::RGB,
        {{VideoBitDepth::_8, {3, 1, 1}},
         {VideoBitDepth::_10, {15, 4, 1}},
         {VideoBitDepth::_12, {9, 2, 1}},
         {VideoBitDepth::_16, {6, 1, 1}},
         {VideoBitDepth::_16f, {6, 1, 1}}}},
    {VideoSampling::YCbCr_4_4_4,
        {{VideoBitDepth::_8, {3, 1, 1}},
         {VideoBitDepth::_10, {15, 4, 1}},
         {VideoBitDepth::_12, {9, 2, 1}},
         {VideoBitDepth::_16, {6, 1, 1}},
         {VideoBitDepth::_16f, {6, 1, 1}}}},
    {VideoSampling::YCbCr_4_2_2,
        {{VideoBitDepth::_8, {4, 2, 1}},
         {VideoBitDepth::_10, {5, 2, 1}},
         {VideoBitDepth::_12, {6, 2, 1}},
         {VideoBitDepth::_16, {8, 2, 1}},
         {VideoBitDepth::_16f, {8, 2, 1}}}},
    {VideoSampling::YCbCr_4_2_0,
        {{VideoBitDepth::_8, {6, 2, 2}},
         {VideoBitDepth::_10, {15, 4, 2}},
         {VideoBitDepth::_12, {9, 2, 2}}}},
    {VideoSampling::KEY,
        {{VideoBitDepth::_8, {1, 1, 1}},
         {VideoBitDepth::_10, {5, 4, 1}},
         {VideoBitDepth::_12, {3, 2, 1}},
         {VideoBitDepth::_16, {2, 1, 1}},
         {VideoBitDepth::_16f, {2, 1, 1}}}}
};

bool ST_2110_20_MediaSettingsCalculator::is_bit_depth_supported(VideoSampling sampling, VideoBitDepth bit_depth)
{
    auto sampling_it = VIDEO_DEPTH_TO_PGROUP_GEOMETRY.find(sampling);
    if (sampling_it != VIDEO_DEPTH_TO_PGROUP_GEOMETRY.end()) {
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

    if (!video_settings.enable_extended_video_formats) {
        if (std::find(SUPPORTED_VIDEO_RESOLUTIONS.begin(), SUPPORTED_VIDEO_RESOLUTIONS.end(),
            video_settings.resolution) == SUPPORTED_VIDEO_RESOLUTIONS.end()) {
            std::cerr << "Error: Unsupported video resolution " << video_settings.resolution
                      << " for this configuration\n";
            return ReturnStatus::failure;
        }
        if (std::find(SUPPORTED_VIDEO_BIT_DEPTHS.begin(), SUPPORTED_VIDEO_BIT_DEPTHS.end(),
            video_settings.bit_depth) == SUPPORTED_VIDEO_BIT_DEPTHS.end()) {
            std::cerr << "Error: Unsupported video bit depth " << enum_to_string(video_settings.bit_depth)
                      << " for this configuration\n";
            return ReturnStatus::failure;
        }
    }

    if (!is_bit_depth_supported(video_settings.sampling_type, video_settings.bit_depth)) {
        std::cerr << "Error: Unsupported bit depth " << enum_to_string(video_settings.bit_depth)
                  << " for " << enum_to_string(video_settings.sampling_type) << " sampling\n";
        return ReturnStatus::failure;
    }

    const auto pgroup =
        VIDEO_DEPTH_TO_PGROUP_GEOMETRY.at(video_settings.sampling_type).at(video_settings.bit_depth);

    if (video_settings.sampling_type == VideoSampling::YCbCr_4_2_0 &&
        video_settings.video_scan_type == VideoScanType::Interlaced) {
        std::cerr << "Error: " << enum_to_string(VideoSampling::YCbCr_4_2_0)
                  << " sampling is not permitted with interlaced video per SMPTE ST 2110-20\n";
        return ReturnStatus::failure;
    }

    if (video_settings.resolution.width > MAX_ST_2110_20_VIDEO_DIMENSION ||
        video_settings.resolution.height > MAX_ST_2110_20_VIDEO_DIMENSION) {
        std::cerr << "Error: Invalid resolution " << video_settings.resolution
                  << ": width and height must each be at most " << MAX_ST_2110_20_VIDEO_DIMENSION << "\n";
        return ReturnStatus::failure;
    }

    /* A media unit must hold whole pgroup rows. */
    if (video_settings.resolution.height % pgroup.rows != 0) {
        std::cerr << "Error: Invalid resolution " << video_settings.resolution << " for "
                  << enum_to_string(video_settings.sampling_type) << " "
                  << enum_to_string(video_settings.video_scan_type)
                  << " video: height must be a multiple of " << pgroup.rows << "\n";
        return ReturnStatus::failure;
    }

    /* Current field handling assumes equal-height fields. */
    if (video_settings.video_scan_type == VideoScanType::Interlaced &&
        video_settings.resolution.height % 2 != 0) {
        std::cerr << "Error: Unsupported interlaced resolution " << video_settings.resolution
                  << ": the current sender requires equal-height fields (even frame height)\n";
        return ReturnStatus::failure;
    }

    const uint32_t bytes_in_pgroup = pgroup.octets;
    const uint32_t pixel_columns_in_pgroup = pgroup.columns;
    const uint32_t pgroups_in_pgroup_row =
        (video_settings.resolution.width + pixel_columns_in_pgroup - 1) / pixel_columns_in_pgroup;
    const uint32_t pgroup_rows_in_frame = video_settings.resolution.height / pgroup.rows;

    video_settings.bytes_per_media_unit =
        static_cast<size_t>(bytes_in_pgroup) * pgroups_in_pgroup_row * pgroup_rows_in_frame;

    const size_t protocol_header_size = RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE;
    const size_t max_raw_payload = max_raw_media_bytes(protocol_header_size);

    uint32_t pgroups_in_packet = 1; /* Non-zero initialization for Coverity to avoid a false "divide by zero" error */
    for (uint32_t pkt_cnt = 1; pkt_cnt <= pgroups_in_pgroup_row; pkt_cnt++) {
        if (pgroups_in_pgroup_row % pkt_cnt != 0) {
            continue;
        }
        /* At pkt_cnt == pgroups_in_pgroup_row, pgroups_in_packet is always non-zero. */
        pgroups_in_packet = pgroups_in_pgroup_row / pkt_cnt;
        if (static_cast<size_t>(pgroups_in_packet) * bytes_in_pgroup <= max_raw_payload) {
            break;
        }
    }

    const size_t raw_payload_size = static_cast<size_t>(pgroups_in_packet) * bytes_in_pgroup;
    if (!fits_standard_udp_size(protocol_header_size, raw_payload_size)) {
        std::cerr << "Error: Video UDP datagram size ("
                  << (UDP_HEADER_SIZE + protocol_header_size + raw_payload_size)
                  << " bytes) exceeds ST 2110-10 Standard UDP Size Limit ("
                  << STANDARD_UDP_SIZE_LIMIT << " bytes)" << std::endl;
        return ReturnStatus::failure;
    }

    video_settings.packets_in_pgroup_row = pgroups_in_pgroup_row / pgroups_in_packet;
    video_settings.protocol_header_size = static_cast<uint16_t>(protocol_header_size);
    video_settings.raw_packet_payload_size = static_cast<uint16_t>(raw_payload_size);
    video_settings.packet_payload_size = static_cast<uint16_t>(
        video_settings.raw_packet_payload_size + video_settings.protocol_header_size);
    if (video_settings.header_data_split) {
        video_settings.packet_app_header_size = video_settings.protocol_header_size;
        video_settings.packet_payload_size -= video_settings.protocol_header_size;
    }

    video_settings.pgroups_in_pgroup_row = pgroups_in_pgroup_row;
    video_settings.pgroups_in_packet = pgroups_in_packet;
    video_settings.pixel_columns_in_pgroup = static_cast<uint16_t>(pixel_columns_in_pgroup);
    video_settings.pixel_rows_in_pgroup = static_cast<uint8_t>(pgroup.rows);
    video_settings.packets_in_media_unit = static_cast<uint32_t>(
        video_settings.packets_in_pgroup_row * pgroup_rows_in_frame);
    video_settings.media_unit_time_interval_ns = rational_cast<double>(NS_IN_SEC / video_settings.frame_rate);
    video_settings.lines_in_media_unit = video_settings.resolution.height;
    video_settings.ticks_per_media_unit = video_settings.sample_rate / video_settings.frame_rate;

    if (video_settings.video_scan_type == VideoScanType::Interlaced) {
        video_settings.packets_in_media_unit /= 2;
        video_settings.lines_in_media_unit /= 2;
        video_settings.media_unit_time_interval_ns /= 2;
        video_settings.ticks_per_media_unit /= 2;
        video_settings.bytes_per_media_unit /= 2;
    }

    if (video_settings.packets_in_media_unit == 0) {
        std::cerr << "Error: Resolution " << video_settings.resolution.width << "x"
                  << video_settings.resolution.height << " holds no complete pgroup row\n";
        return ReturnStatus::failure;
    }

    /* A chunk beyond the media unit only reserves memory that no packet can reach. */
    if (video_settings.packets_in_chunk > video_settings.packets_in_media_unit) {
        std::cerr << "Warning: Custom chunk size (" << video_settings.packets_in_chunk
            << ") is reduced to the packets in media unit ("
            << video_settings.packets_in_media_unit << ")" << std::endl;
        video_settings.packets_in_chunk = video_settings.packets_in_media_unit;
    }

    if (video_settings.packets_in_chunk) {
        std::cout << "Using custom chunk size: " << video_settings.packets_in_chunk << std::endl;
    } else {
        constexpr size_t max_pgroup_rows_in_chunk = 4;
        const size_t pgroup_rows_in_media_unit =
            video_settings.packets_in_media_unit / video_settings.packets_in_pgroup_row;
        const size_t pgroup_rows_in_chunk = std::min({ max_pgroup_rows_in_chunk, pgroup_rows_in_media_unit,
            MAX_DEFAULT_PACKETS_IN_CHUNK / video_settings.packets_in_pgroup_row });
        /* A single pgroup row above the chunk limit is split across chunks. */
        video_settings.packets_in_chunk = pgroup_rows_in_chunk ?
            pgroup_rows_in_chunk * video_settings.packets_in_pgroup_row :
            std::min<size_t>(video_settings.packets_in_media_unit, MAX_DEFAULT_PACKETS_IN_CHUNK);
    }

    /* The final chunk of a media unit holds the remainder, its reserved slots stay empty. */
    video_settings.chunks_in_media_unit =
        (video_settings.packets_in_media_unit + video_settings.packets_in_chunk - 1) / video_settings.packets_in_chunk;
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
