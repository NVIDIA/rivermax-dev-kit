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

#ifndef RDK_SERVICES_MEDIA_MEDIA_SETTINGS_VIDEO_H_
#define RDK_SERVICES_MEDIA_MEDIA_SETTINGS_VIDEO_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "rdk/services/media/media_calc_interface.h"
#include "rdk/services/media/media_settings.h"
#include "rdk/services/sdp/sdp_common_descriptions.h"
#include "rdk/services/settings/app_settings.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 * @brief: SMPTE 2110-20 video media settings.
 *
 * The struct will be used to hold media parameters for
 * SMPTE 2110-20 video streams.
 */
struct SMPTE_2110_20_MediaSettings : public MediaSettings
{
    /**
     * @brief: Default constructor.
     */
    SMPTE_2110_20_MediaSettings() = default;
    /**
     * @brief: Constructor from AppSettings.
     *
     * @param [in] app_settings: Application settings to configure from.
     * @param [in] is_alpha_stream: Whether this is an alpha/key stream (default: false).
     */
    explicit SMPTE_2110_20_MediaSettings(const AppSettings& app_settings, bool is_alpha_stream = false);
    virtual ~SMPTE_2110_20_MediaSettings() = default;
    virtual SMPTEStandard get_smpte_standard() const override { return SMPTEStandard::ST_2110_20; };
    /**
     * @brief: Returns packets per frame for Rivermax API.
     *
     * Multiplies by 2 for interlaced since Rivermax expects full frame packets.
     *
     * @return: Packets per full frame.
     */
    virtual size_t get_packets_per_frame() const override {
        return video_scan_type == VideoScanType::Interlaced
            ? packets_in_media_unit * 2
            : packets_in_media_unit;
    }

    Resolution resolution = { _1080_WIDTH, _1080_HEIGHT };
    FrameRate frame_rate = { 60 };
    VideoSampling sampling_type = VideoSampling::YCbCr_4_2_2;
    VideoBitDepth bit_depth = VideoBitDepth::_10;
    VideoScanType video_scan_type = VideoScanType::Progressive;
    Colorimetry colorimetry = Colorimetry::BT709;
    SMPTEStandardNumber smpte_standard_number = SMPTEStandardNumber::ST2110_20_2017;
    size_t packets_in_line = 0;
    size_t lines_in_frame_field = 0;
    uint16_t pixels_per_packet = 0;
};

}  // namespace services
}  // namespace dev_kit
}  // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_MEDIA_SETTINGS_VIDEO_H_ */
