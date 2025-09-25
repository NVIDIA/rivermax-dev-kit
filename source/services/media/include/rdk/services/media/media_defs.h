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

#ifndef RDK_SERVICES_MEDIA_MEDIA_DEFS_H_
#define RDK_SERVICES_MEDIA_MEDIA_DEFS_H_

#include <cstdint>
#include <string>
#include <stdexcept>
#include <iostream>
#include <ostream>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <memory>

#include "rdk/services/media/media_calc_interface.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/sdp/sdp_defs.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 * @brief: Video frame rate.
 */
struct FrameRate
{
    uint16_t num;
    uint16_t denom;

    /**
     * @brief: Video frame rate default constructor.
     */
    FrameRate() : num(0), denom(0) {}
    /**
    * @brief: Video frame rate constructor.
    *
    * @param [in] num: The numerator of the frame rate.
    * @param [in] denom: The denominator of the frame rate.
    */
    FrameRate(uint16_t num, uint16_t denom) : num(num), denom(denom) {}
    /**
     * @brief: Video frame rate default constructor.
     *
     * @param [in] num: The numerator of the frame rate.
     */
     FrameRate(uint16_t num) : num(num), denom(1) {}
    /**
     * @brief: Video frame rate constructor.
     *
     * @param [in] frame_rate: The frame rate in the format <numerator>/<denominator> or <integer>.
     */
    FrameRate(const std::string& frame_rate)
    {
        auto is_number = [](const std::string& str) {
            return str.find_first_not_of("0123456789") == std::string::npos;
        };

        size_t slash_position = frame_rate.find('/');
        if (slash_position != std::string::npos) {
            std::string num_str = frame_rate.substr(0, slash_position);
            std::string denom_str = frame_rate.substr(slash_position + 1);
            if (is_number(num_str) && is_number(denom_str)) {
                num = static_cast<uint16_t>(std::stoi(num_str));
                denom = static_cast<uint16_t>(std::stoi(denom_str));
                return;
            }
        } else if (is_number(frame_rate)) {
            num = static_cast<uint16_t>(std::stoi(frame_rate));
            denom = 1;
            return;
        }
        throw std::invalid_argument("Invalid frame rate format. Expected <numerator>/<denominator> or <integer>");
    }
    /**
     * @brief: Converts the frame rate to a string.
     *
     * @return: The string representation of the frame rate.
     */
    operator std::string() const
    {
        return (denom == 1) ? std::to_string(num) : std::to_string(num) + "/" + std::to_string(denom);
    }
    /**
     * @brief: Output stream operator.
     *
     * @param [in] os: The output stream.
     * @param [in] frame_rate: The frame rate.
     *
     * @return: The output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const FrameRate& frame_rate) {
        os << std::string(frame_rate);
        return os;
    }
    /**
     * @brief: Equality operator.
     *
     * @param [in] other: The other frame rate.
     *
     * @return: True if the frame rates are equal, false otherwise.
     */
    bool operator==(const FrameRate& other) const { return num == other.num && denom == other.denom; }
    /**
     * @brief: Inequality operator.
     *
     * @param [in] other: The other frame rate.
     *
     * @return: True if the frame rates are not equal, false otherwise.
     */
    bool operator!=(const FrameRate& other) const { return !(*this == other); }
};
/**
 * @brief: Video Resolution.
 */
struct Resolution
{
    uint16_t width;
    uint16_t height;

    /**
     * @brief: Video resolution default constructor.
     */
    Resolution() : width(0), height(0) {}
    /**
     * @brief: Video resolution constructor.
     *
     * @param [in] width: The width of the resolution.
     * @param [in] height: The height of the resolution.
     */
    Resolution(uint16_t width, uint16_t height) : width(width), height(height) {}
    /**
     * @brief: Video resolution constructor.
     *
     * @param [in] resolution: The resolution in the format <width>x<height>.
     */
    Resolution(const std::string& resolution)
    {
        auto is_number = [](const std::string& str) {
            return str.find_first_not_of("0123456789") == std::string::npos;
        };

        size_t x_position = resolution.find('x');
        if (x_position != std::string::npos) {
            std::string width_str = resolution.substr(0, x_position);
            std::string height_str = resolution.substr(x_position + 1);
            if (is_number(width_str) && is_number(height_str)) {
                width = static_cast<uint16_t>(std::stoi(width_str));
                height = static_cast<uint16_t>(std::stoi(height_str));
                return;
            }
        }
        throw std::invalid_argument("Invalid resolution format. Expected <width>x<height>");
    }
    /**
     * @brief: Converts the resolution to a string.
     *
     * @return: The string representation of the resolution in the format <width>x<height>.
     */
    operator std::string() const { return std::to_string(width) + "x" + std::to_string(height); }
    /**
     * @brief: Output stream operator.
     *
     * @param [in] os: The output stream.
     * @param [in] resolution: The resolution.
     *
     * @return: The output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const Resolution& resolution) {
        os << std::string(resolution);
        return os;
    }
    /**
     * @brief: Equality operator.
     *
     * @param [in] other: The other resolution.
     *
     * @return: True if the resolutions are equal, false otherwise.
     */
    bool operator==(const Resolution& other) const { return width == other.width && height == other.height; }
    /**
     * @brief: Inequality operator.
     *
     * @param [in] other: The other resolution.
     *
     * @return: True if the resolutions are not equal, false otherwise.
     */
    bool operator!=(const Resolution& other) const { return !(*this == other); }
};

/* Time constants */
constexpr size_t NS_IN_SEC = std::chrono::nanoseconds{ std::chrono::seconds{ 1 } }.count();
constexpr size_t NS_IN_USEC = std::chrono::nanoseconds{ std::chrono::microseconds{ 1 } }.count();
constexpr size_t NS_IN_MSEC = std::chrono::nanoseconds{ std::chrono::milliseconds{ 1 } }.count();
constexpr uint8_t LEAP_SECONDS = 37;
/* Resolution constants */
constexpr size_t FHD_WIDTH = 1920;
constexpr size_t FHD_HEIGHT = 1080;
constexpr size_t UHD_WIDTH = 3840;
constexpr size_t UHD_HEIGHT = 2160;
/* RTP header constants */
constexpr size_t RTP_HEADER_EXT_SEQ_NUM_SIZE = 2;
constexpr size_t RTP_HEADER_SRD_MIN_SIZE = RTP_HEADER_EXT_SEQ_NUM_SIZE + 2;  // When first SRD length is 0
constexpr size_t RTP_HEADER_SRD_SIZE = 6;
constexpr size_t RTP_HEADER_SIZE = 12;
constexpr size_t RTP_HEADER_SMPTE_2110_20_MAX_SRDS_NUM = 3;
constexpr size_t RTP_HEADER_MAX_CSRCS = 15;
constexpr size_t RTP_HEADER_CSRC_GRANULARITY_BYTES = 4;
constexpr size_t RTP_SINGLE_SRD_HEADER_SIZE = RTP_HEADER_SRD_SIZE + RTP_HEADER_EXT_SEQ_NUM_SIZE;
constexpr size_t RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE = \
    RTP_HEADER_SIZE + RTP_SINGLE_SRD_HEADER_SIZE;
constexpr uint32_t RTP_SEQUENCE_NUMBER_MASK_16BIT = 0xFFFF;
constexpr uint32_t RTP_SEQUENCE_NUMBER_MASK_32BIT = 0xFFFFFFFF;
constexpr uint8_t RTP_VERSION_MASK = 0xC0;
constexpr uint8_t RTP_VERSION_2 = 0x80;
constexpr uint8_t RTP_M_BIT_MASK = 0x80;
/* Video constants */
constexpr size_t VIDEO_TRO_DEFAULT_MODIFICATION = 2;
/* Supported video resolutions */
const std::vector<Resolution> SUPPORTED_VIDEO_RESOLUTIONS = {
    { FHD_WIDTH, FHD_HEIGHT },
    { UHD_WIDTH, UHD_HEIGHT },
    { FHD_HEIGHT, FHD_WIDTH },
    { UHD_HEIGHT, UHD_WIDTH }
};
/* Supported video frame rates */
const std::vector<FrameRate> SUPPORTED_VIDEO_FRAME_RATES = {
    { 24 },
    { 25 },
    { 30 },
    { 50 },
    { 60 }
};
/* Supported video sampling types */
const std::vector<VideoSampling> SUPPORTED_VIDEO_SAMPLING_TYPES = {
    VideoSampling::YCbCr_4_4_4,
    VideoSampling::YCbCr_4_2_2,
    VideoSampling::YCbCr_4_2_0,
    VideoSampling::RGB
};
/**
 * @brief: Enum class for SMPTE Media content types.
 */
enum class SMPTEStandard
{
    ST_2110_20,
    ST_2110_30,
    ST_2110_40,
    Unknown
};
 /* Supported video bit depths */
const std::vector<VideoBitDepth> SUPPORTED_VIDEO_BIT_DEPTHS = {
    VideoBitDepth::_8,
    VideoBitDepth::_10,
    VideoBitDepth::_12
};
/**
 * @brief: Application media related settings.
 *
 * The struct will be used to hold application media parameters required
 * for the application to operate as requested by the user.
 */
struct AppMediaSettings
{
    std::string sdp;
    uint32_t media_block_index = 0;
    FrameRate frame_rate;
    Resolution resolution = { FHD_WIDTH, FHD_HEIGHT };
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
    uint8_t payload_type = 96;
    uint16_t pixels_per_packet = 0;
};
/**
 * @brief: Media settings calculator interface.
 *
 * This interface provides methods for calculating media parameters for different media types.
 */
 class IMediaSettingsCalculator;
/**
 * @brief: Media settings.
 *
 * The struct will be used to hold media parameters for
 * all different media types processed by the application.
 */
struct MediaSettings
{
    virtual ~MediaSettings() = default;

    static constexpr size_t MIN_FRAMES_FOR_SIMULTANEOUS_TX_AND_FILLUP = 2;
    static constexpr size_t DEFAULT_NUM_OF_FRAMES_IN_MEM_BLOCK = 10;
    static constexpr size_t DEFAULT_NUM_OF_MEM_BLOCKS = 1;
    static constexpr uint8_t DEFAULT_PTP_DOMAIN_ID = 127;
    static constexpr uint8_t DEFAULT_PAYLOAD_TYPE = 96;

    uint32_t sdp_media_block_index = 0;
    size_t sample_rate = 90000;
    uint32_t packets_in_frame_field = 0;
    double frame_field_time_interval_ns = 0;
    double ticks_per_frame = 0;
    size_t chunks_in_frame_field = 0;
    size_t packets_in_chunk = 0;
    size_t frames_fields_in_mem_block = 0;
    /**
     * @brief: Reference clock ID.
     *
     * The reference clock ID is used to identify the reference clock.
     * If the reference clock is PTP, the ID is either the PTP grandmaster clock identity,
     * or the reference clock source is defined as traceable (if the ID is empty).
     * If the reference clock is not PTP, the ID is the local MAC address.
     */
    std::string refclk_id = "";
    /**
     * @brief: Reference clock is PTP.
     */
    bool ref_clk_is_ptp = true;
    uint8_t ptp_domain_id = DEFAULT_PTP_DOMAIN_ID;
    size_t bytes_per_frame = 0;
    uint16_t protocol_header_size = 0;
    bool header_data_split = false;
    uint16_t packet_app_header_size = 0;
    uint16_t packet_payload_size = 0;
    uint16_t raw_packet_payload_size = 0;
    uint8_t payload_type = DEFAULT_PAYLOAD_TYPE;
    size_t chunks_in_mem_block = 0;
    size_t packets_in_mem_block = 0;
    size_t requested_num_of_mem_blocks = DEFAULT_NUM_OF_MEM_BLOCKS;
    size_t data_stride_size = 0;
    size_t app_header_stride_size = 0;
    /**
     * @brief: Returns the media type.
     *
     * @return: Media type.
     */
    virtual SMPTEStandard get_media_type() const = 0;
    std::shared_ptr<IMediaSettingsCalculator> media_settings_calculator;
};
/**
 * @brief: SMPTE 2110-20 video media settings.
 *
 * The struct will be used to hold media parameters for
 * SMPTE 2110-20 video streams.
 */
 struct SMPTE_2110_20_MediaSettings : public MediaSettings
{
    virtual ~SMPTE_2110_20_MediaSettings() = default;
    virtual SMPTEStandard get_media_type() const override { return SMPTEStandard::ST_2110_20; };
    Resolution resolution = { FHD_WIDTH, FHD_HEIGHT };
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

#endif /* RDK_SERVICES_MEDIA_MEDIA_DEFS_H_ */
