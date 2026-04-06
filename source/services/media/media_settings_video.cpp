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

#include "rdk/services/media/media_settings_video.h"

using namespace rdk::services;

SMPTE_2110_20_MediaSettings::SMPTE_2110_20_MediaSettings(const AppSettings& app_settings, bool is_alpha_stream)
{
    // Initialize base MediaSettings common fields
    header_data_split = app_settings.header_data_split;
    requested_num_of_mem_blocks = MediaSettings::DEFAULT_NUM_OF_MEM_BLOCKS;
    media_units_in_mem_block = app_settings.media.frames_fields_in_mem_block;
    payload_type = PAYLOAD_TYPE_ST_2110_20;

    // Initialize ref clock settings
    ref_clk_is_ptp = app_settings.ref_clk_is_ptp;

    // Initialize packets in chunk if specified
    if (app_settings.num_of_packets_in_chunk_specified) {
        packets_in_chunk = app_settings.num_of_packets_in_chunk;
    }

    // Initialize SMPTE 2110-20 specific fields
    resolution = app_settings.media.resolution;
    frame_rate = app_settings.media.frame_rate;
    bit_depth = is_alpha_stream ? app_settings.media.alpha_bit_depth : app_settings.media.color_bit_depth;
    video_scan_type = app_settings.media.video_scan_type;

    if (is_alpha_stream) {
        sampling_type = VideoSampling::KEY;
        colorimetry = Colorimetry::ALPHA;
        smpte_standard_number = SMPTEStandardNumber::ST2110_20_2021;
        media_file = "";
        dynamic_media_file_load = false;
    } else {
        sampling_type = app_settings.media.sampling_type;
        colorimetry = Colorimetry::BT709;
        smpte_standard_number = SMPTEStandardNumber::ST2110_20_2017;
        media_file = app_settings.video_file;
        dynamic_media_file_load = app_settings.dynamic_media_file_load;
    }
}
