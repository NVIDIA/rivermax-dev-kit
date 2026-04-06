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

#ifndef RDK_SERVICES_MEDIA_VIDEO_SETTINGS_CALCULATOR_H_
#define RDK_SERVICES_MEDIA_VIDEO_SETTINGS_CALCULATOR_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/media/media_settings.h"
#include "rdk/services/media/media_settings_calculator.h"
#include "rdk/services/media/media_settings_video.h"
#include "rdk/services/sdp/sdp.h"

namespace rdk
{
namespace services
{

/**
 * @brief: SMPTE 2110-20 video media settings calculator.
 *
 * This class implements the media settings calculator for SMPTE 2110-20 video streams.
 * It provides methods to calculate video-specific media settings and compose SDP descriptions
 * according to the SMPTE 2110-20 standard.
 */
class ST_2110_20_MediaSettingsCalculator : public MediaSettingsCalculator<SMPTE_2110_20_MediaSettings>
{
public:
    /**
     * @brief: ST_2110_20_MediaSettingsCalculator constructor.
     *
     * @param [in] video_settings: Reference to the video settings to configure.
     * @param [in] extra_parameters: Optional vector of format-specific parameters.
     */
    ST_2110_20_MediaSettingsCalculator(SMPTE_2110_20_MediaSettings& video_settings, const std::vector<FormatSpecificParameter>& extra_parameters = {}) :
        MediaSettingsCalculator<SMPTE_2110_20_MediaSettings>(video_settings, extra_parameters) {}
    /**
     * @brief: Virtual destructor.
     */
    virtual ~ST_2110_20_MediaSettingsCalculator() = default;
    /**
     * @brief: Calculates media settings for SMPTE 2110-20 video.
     *
     * This method performs all necessary calculations to configure the media settings
     * for a SMPTE 2110-20 video stream, including timing, packetization, and memory layout.
     *
     * @return: Status of the calculation operation.
     */
    virtual ReturnStatus calculate_media_settings() override;
    /**
     * @brief: Generates SDP description for SMPTE 2110-20 video.
     *
     * This method generates a Session Description Protocol (SDP) description string
     * for the SMPTE 2110-20 video stream with the specified network parameters.
     * For multiple flows, includes group description and media IDs for SMPTE 2022-7 duplication.
     *
     * @param [in] flows: Vector of flow info containing source IP, destination IP and port.
     *
     * @return: SDP description string for the video stream.
     */
    virtual std::string generate_media_sdp(const std::vector<NetworkFlow>& flows) override;
    /**
     * @brief: Returns the SMPTE standard name.
     *
     * @return: String representation of the SMPTE 2110-20 video SMPTE standard.
     */
    virtual std::string get_smpte_standard_name() const override;
    virtual double align_time_to_media_unit_boundary_ns(uint64_t desired_time_ns) const override;
    /**
     * @brief: Returns the transmit offset value in nanoseconds.
     *
     * @return: Transmit offset value in nanoseconds.
     */
    virtual double get_transmit_offset_ns() const override;
    /**
     * @brief: Calculates TRO and TRS timing parameters.
     *
     * This method calculates the TRO (Time Read Offset)
     * and TRS (Time Read Spacing) timing parameters
     * according to SMPTE ST2110-20.
     *
     * @param [out] tro: The calculated TRO (Time Read Offset) value.
     * @param [out] trs: The calculated TRS (Time Read Spacing) value.
     */
    void calculate_tro_trs(double& tro, double& trs) const;
    /**
     * @brief: Check if the given sampling type and bit depth are supported.
     *
     * @param [in] sampling: The video sampling type.
     * @param [in] bit_depth: The color bit depth.
     *
     * @return: True if the sampling type and bit depth are supported, false otherwise.
     */
    static bool is_bit_depth_supported(VideoSampling sampling, VideoBitDepth bit_depth);
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_MEDIA_VIDEO_SETTINGS_CALCULATOR_H_ */
