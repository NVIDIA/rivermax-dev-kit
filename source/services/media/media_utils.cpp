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

#include <cstdint>
#include <unordered_map>

#include "rdk/services/media/media_defs.h"
#include "rdk/services/media/media_utils.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/sdp/sdp.h"

using namespace rivermax::dev_kit::services;

using BytesPerPixelRatio = std::pair<uint32_t, uint32_t>;
using ColorDepthPixelRatioMap =
    std::unordered_map<VideoSampling, std::unordered_map<ColorBitDepth, BytesPerPixelRatio>>;
/**
 * @brief: Map of video sampling type to pixel ratio for each color bit depth.
 */
const ColorDepthPixelRatioMap COLOR_DEPTH_TO_PIXEL_RATIO = {
    {VideoSampling::RGB,
     {{ColorBitDepth::_8, {3, 1}},
      {ColorBitDepth::_10, {15, 4}},
      {ColorBitDepth::_12, {9, 2}}}},
    {VideoSampling::YCbCr_4_4_4,
     {{ColorBitDepth::_8, {3, 1}},
      {ColorBitDepth::_10, {15, 4}},
      {ColorBitDepth::_12, {9, 2}}}},
    {VideoSampling::YCbCr_4_2_2,
     {{ColorBitDepth::_8, {4, 2}},
      {ColorBitDepth::_10, {5, 2}},
      {ColorBitDepth::_12, {6, 2}}}},
    {VideoSampling::YCbCr_4_2_0,
     {{ColorBitDepth::_8, {6, 4}},
      {ColorBitDepth::_10, {15, 8}},
      {ColorBitDepth::_12, {9, 4}}}}
};
/**
 * @brief: Map of video sampling type and color bit depth to payload size.
 */
const std::unordered_map<VideoSampling, std::unordered_map<ColorBitDepth, size_t>> COLOR_DEPTH_TO_PAYLOAD_SIZE = {
    {VideoSampling::RGB,
     {{ColorBitDepth::_8, 1440 + RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE},
      {ColorBitDepth::_10, 1440 + RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE},
      {ColorBitDepth::_12, 1440 + RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE}}},
    {VideoSampling::YCbCr_4_4_4,
     {{ColorBitDepth::_8, 1440 + RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE},
      {ColorBitDepth::_10, 1440 + RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE},
      {ColorBitDepth::_12, 1440 + RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE}}},
    {VideoSampling::YCbCr_4_2_2,
     {{ColorBitDepth::_8, 1280 + RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE},
      {ColorBitDepth::_10, 1200 + RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE},
      {ColorBitDepth::_12, 1080 + RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE}}},
    {VideoSampling::YCbCr_4_2_0,
     {{ColorBitDepth::_8, 1440 + RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE},
      {ColorBitDepth::_10, 1200 + RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE},
      {ColorBitDepth::_12, 1080 + RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE}}}
};
/**
 * @brief: Check if the given sampling type and bit depth are supported.
 *
 * @param [in] sampling: The video sampling type.
 * @param [in] bit_depth: The color bit depth.
 *
 * @return: True if the sampling type and bit depth are supported, false otherwise.
 */
bool is_bit_depth_supported(VideoSampling sampling, ColorBitDepth bit_depth)
{
    auto sampling_it = COLOR_DEPTH_TO_PIXEL_RATIO.find(sampling);
    if (sampling_it != COLOR_DEPTH_TO_PIXEL_RATIO.end()) {
        return sampling_it->second.find(bit_depth) != sampling_it->second.end();
    }
    return false;
}
/**
 * @brief: Initialize common media settings.
 *
 * @param [in,out] settings: The application settings.
 *
 * @return: The return status.
 */
static ReturnStatus initialize_common_media_settings(AppSettings& settings)
{
    auto& s = settings;

    s.num_of_total_flows = s.num_of_total_streams;
    s.num_of_memory_blocks = 1;

    uint32_t num_of_pixels = s.media.resolution.width * s.media.resolution.height;

    float bytes_per_pixel;
    if (is_bit_depth_supported(s.media.sampling_type, s.media.bit_depth)) {
        auto bytes_per_pixel_ratio = COLOR_DEPTH_TO_PIXEL_RATIO.at(s.media.sampling_type).at(s.media.bit_depth);
        uint32_t bytes = bytes_per_pixel_ratio.first;
        uint32_t pixels = bytes_per_pixel_ratio.second;
        bytes_per_pixel = static_cast<float>(bytes) / static_cast<float>(pixels);
        s.media.bytes_per_frame = static_cast<uint32_t>(num_of_pixels * (bytes / static_cast<double>(pixels)));
        s.packet_payload_size =
            static_cast<uint16_t>(COLOR_DEPTH_TO_PAYLOAD_SIZE.at(s.media.sampling_type).at(s.media.bit_depth));
    } else {
        std::cerr << "Unsupported sampling type or bit depth: " << enum_to_string(s.media.sampling_type) << ", "
                  << enum_to_string(s.media.bit_depth) << "bit \n";
        return ReturnStatus::failure;
    }

    s.media.protocol_header_size = RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE;
    s.media.raw_packet_payload_size = s.packet_payload_size - RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE;
    if (s.header_data_split) {
        s.packet_app_header_size = RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE;
        s.packet_payload_size -= RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE;
    }

    s.media.pixels_per_packet = static_cast<uint16_t>(s.media.raw_packet_payload_size / bytes_per_pixel);
    s.media.packets_in_frame_field = static_cast<uint32_t>(s.media.bytes_per_frame / s.media.raw_packet_payload_size);
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
        constexpr size_t lines_in_chunk = 4;
        s.num_of_packets_in_chunk = lines_in_chunk * s.media.packets_in_line;
    }

    s.media.frame_field_time_interval_ns =
        NS_IN_SEC / static_cast<double>(s.media.frame_rate.num) / s.media.frame_rate.denom;
    s.media.lines_in_frame_field = s.media.resolution.height;

    s.media.ticks_per_frame =
        (s.media.sample_rate / (s.media.frame_rate.num / static_cast<double>(s.media.frame_rate.denom)));

    if (s.media.video_scan_type == VideoScanType::Interlaced) {
        s.media.packets_in_frame_field /= 2;
        s.media.lines_in_frame_field /= 2;
        s.media.frame_field_time_interval_ns /= 2;
        s.media.ticks_per_frame /= 2;
    }

    s.media.chunks_in_frame_field =
        static_cast<size_t>(std::ceil(s.media.packets_in_frame_field / static_cast<double>(s.num_of_packets_in_chunk)));
    s.num_of_chunks_in_mem_block = s.media.frames_fields_in_mem_block * s.media.chunks_in_frame_field;
    s.num_of_packets_in_mem_block = s.num_of_chunks_in_mem_block * s.num_of_packets_in_chunk;

    return ReturnStatus::success;
}
/**
 * @brief: Initialize media SDP.
 *
 * @param [in] settings: The application settings.
 * @param [in] extra_format_specific_parameters: The extra format specific parameters (default = empty).
 */
static void initialize_media_sdp(
    AppSettings& settings, const std::vector<FormatSpecificParameter>& extra_format_specific_parameters /*= {}*/
)
{
    auto& s = settings;

    auto session_description = SessionDescription::Builder(s.local_ip)
        .set_session_id(SDPManager::generate_ntp_id())
        .set_session_version(SDPManager::generate_ntp_id() + 1)
        .set_session_name("SMPTE ST2110-20")
        .build();

    auto time_description = TimeDescription::Builder().build();

    auto media_description = SMPTE2110_20_MediaDescription::Builder(
        s.destination_port, TransportProtocol::RTP_AVP, "96", s.destination_ip)
        .set_source_filter(SourceFilterAttribute::Builder(s.destination_ip, s.local_ip).build())
        .set_sampling(s.media.sampling_type)
        .set_width(s.media.resolution.width)
        .set_height(s.media.resolution.height)
        .set_exact_frame_rate(s.media.frame_rate)
        .set_depth(s.media.bit_depth)
        .set_video_scan_type(s.media.video_scan_type)
        .set_timestamp_ref_clock(s.ref_clk_is_ptp || s.local_mac.empty() ? TimestampRefClock::PTP : TimestampRefClock::LocalMAC)
        .set_timestamp_ref_clock_ptp_traceable(s.ref_clk_is_ptp || s.local_mac.empty())
        .set_timestamp_ref_clock_local_mac(s.local_mac)
        .set_extra_format_specific_parameters(extra_format_specific_parameters)
        .build();

    s.media.sdp = SDPManager::Builder(std::move(session_description), std::move(time_description))
        .add_media_description(std::move(media_description))
        .build()->to_string();
}

ReturnStatus rivermax::dev_kit::services::initialize_media_settings(
    AppSettings& settings, const std::vector<FormatSpecificParameter>& extra_format_specific_parameters /*= {}*/
)
{
    auto status = initialize_common_media_settings(settings);
    if (status != ReturnStatus::success) {
        std::cerr << "Failed to initialize common media settings" << std::endl;
        return status;
    }
    initialize_media_sdp(settings, extra_format_specific_parameters);

    return ReturnStatus::success;
}

void rivermax::dev_kit::services::calculate_tro_trs(MediaSettings& media_settings, double& tro, double& trs)
{
    auto& s = media_settings;
    double t_frame_ns;
    double r_active;
    double tro_default_multiplier;

    if (s.video_scan_type == VideoScanType::Progressive) {
        t_frame_ns = s.frame_field_time_interval_ns;
    } else {
        t_frame_ns = s.frame_field_time_interval_ns * 2;
    }

    if (s.video_scan_type == VideoScanType::Progressive) {
        r_active = (1080.0 / 1125.0);
        if (s.resolution.height >= FHD_HEIGHT) { // As defined by SMPTE 2110-21 6.3.2
            tro_default_multiplier = (43.0 / 1125.0);
        } else {
            tro_default_multiplier = (28.0 / 750.0);
        }
    } else {
        if (s.resolution.height >= FHD_HEIGHT) { // As defined by SMPTE 2110-21 6.3.3
            r_active = (1080.0 / 1125.0);
            tro_default_multiplier = (22.0 / 1125.0);
        } else if (s.resolution.height >= 576) {
            r_active = (576.0 / 625.0);
            tro_default_multiplier = (26.0 / 625.0);
        } else {
            r_active = (487.0 / 525.0);
            tro_default_multiplier = (20.0 / 525.0);
        }
    }

    uint32_t packets_in_frame;

    if (s.video_scan_type == VideoScanType::Progressive) {
        packets_in_frame = s.packets_in_frame_field;
    } else {
        packets_in_frame = s.packets_in_frame_field * 2;
    }

    trs = (t_frame_ns * r_active) / packets_in_frame;
    tro = (tro_default_multiplier * t_frame_ns) - (VIDEO_TRO_DEFAULT_MODIFICATION * trs);
}
