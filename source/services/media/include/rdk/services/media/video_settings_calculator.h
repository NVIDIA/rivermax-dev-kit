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

namespace rivermax
{
namespace dev_kit
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
     *
     * @param [in] source_ip: Source IP address for the stream.
     * @param [in] source_port: Source port number for the stream.
     * @param [in] destination_ip: Destination IP address for the stream.
     * @param [in] destination_port: Destination port number for the stream.
     *
     * @return: SDP description string for the video stream.
     */
    virtual std::string generate_media_sdp(const std::string& source_ip, const uint16_t source_port,
        const std::string& destination_ip, const uint16_t destination_port) override;
    /**
     * @brief: Generates SDP description for SMPTE 2110-20 video with duplicate media.
     *
     * This method generates a Session Description Protocol (SDP) description string
     * for the SMPTE 2110-20 video stream with the specified network parameters.
     *
     * @param [in] source_ip_a: Source IP address for the first stream.
     * @param [in] source_port_a: Source port number for the first stream.
     * @param [in] destination_ip_a: Destination IP address for the first stream.
     * @param [in] destination_port_a: Destination port number for the first stream.
     * @param [in] source_ip_b: Source IP address for the second stream.
     * @param [in] source_port_b: Source port number for the second stream.
     * @param [in] destination_ip_b: Destination IP address for the second stream.
     * @param [in] destination_port_b: Destination port number for the second stream.
     *
     * @return: SDP description string for the video stream.
     */
    virtual std::string generate_media_dup_sdp(const std::string& source_ip_a, const uint16_t source_port_a,
        const std::string& destination_ip_a, const uint16_t destination_port_a,
        const std::string& source_ip_b, const uint16_t source_port_b,
        const std::string& destination_ip_b, const uint16_t destination_port_b) override;
    

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
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_VIDEO_SETTINGS_CALCULATOR_H_ */
