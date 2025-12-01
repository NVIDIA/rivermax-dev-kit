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

#ifndef RDK_SERVICES_MEDIA_APP_MEDIA_SETTINGS_H_
#define RDK_SERVICES_MEDIA_APP_MEDIA_SETTINGS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "rdk/services/media/media_defs.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 * @brief: Application level media settings.
 *
 * The struct will be used to hold application media parameters required
 * for the application to operate as requested by the user.
 */
struct AppMediaSettings
{
    std::string sdp;
    uint32_t media_block_index = 0;
    FrameRate frame_rate;
    Resolution resolution = { _1080_WIDTH, _1080_HEIGHT };
    VideoSampling sampling_type = VideoSampling::YCbCr_4_2_2;
    VideoBitDepth color_bit_depth = VideoBitDepth::_10;
    VideoBitDepth alpha_bit_depth = VideoBitDepth::Unknown;
    bool enable_alpha = false;
    VideoScanType video_scan_type = VideoScanType::Progressive;
    SenderType sender_type = SenderType::_2110TPN;
    size_t sample_rate = 90000;
    uint32_t packets_in_frame_field = 0;
    size_t packets_in_line = 0;
    double frame_field_time_interval_ns = 0;
    double ticks_per_frame = 0;
    size_t lines_in_frame_field = 0;
    size_t chunks_in_frame_field = 0;
    size_t frames_fields_in_mem_block = 0;
    std::string refclk;
    size_t bytes_per_frame = 0;
    size_t protocol_header_size = 0;
    size_t packet_payload_size = 0;
    size_t raw_packet_payload_size = 0;
    uint16_t pixels_per_packet = 0;
    bool enable_video = false;
    bool enable_audio = false;
    bool enable_ancillary = false;
    uint32_t ptime_us = 1000;
    AudioSamplingRate audio_sampling_rate = AudioSamplingRate::_48000;
    AudioEncoding audio_encoding = AudioEncoding::L24;
    uint8_t audio_channels_num = 2;
    std::vector<AncillaryDataIdentifier> ancillary_data_identifiers;
    uint16_t ancillary_data_words_count = DEFAULT_ANCILLARY_DATA_WORDS_COUNT;
};

}  // namespace services
}  // namespace dev_kit
}  // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_APP_MEDIA_SETTINGS_H_ */
