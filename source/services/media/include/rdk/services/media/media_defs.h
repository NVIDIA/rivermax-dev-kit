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

#ifndef RDK_SERVICES_MEDIA_MEDIA_DEFS_H_
#define RDK_SERVICES_MEDIA_MEDIA_DEFS_H_

#include <chrono>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rdk/services/sdp/sdp_defs.h"
#include "rdk/services/utils/rational.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 * @brief: Video frame rate type alias.
 *
 * FrameRate is a type alias for Rational, representing video frame rates
 * as rational numbers (e.g., 60000/1001 for 59.94 fps).
 */
using FrameRate = Rational;
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
/**
 * @brief: Ancillary data identifier (DID/SDID pair).
 */
 struct AncillaryDataIdentifier
 {
     uint8_t did;
     uint8_t sdid;

     bool operator==(const AncillaryDataIdentifier& other) const
     {
         return did == other.did && sdid == other.sdid;
     }
     bool operator!=(const AncillaryDataIdentifier& other) const
     {
         return !(*this == other);
     }
     /**
      * @brief: Output stream operator to print identifier.
      *
      * Format: "0xDID:0xSDID" (e.g. "0x60:0x60")
      *
      * @param [in] os: The output stream.
      * @param [in] id: The ancillary data identifier.
      *
      * @return: The output stream.
      */
     friend std::ostream& operator<<(std::ostream& os, const AncillaryDataIdentifier& id)
     {
         os << "0x" << std::hex << static_cast<int>(id.did) << ":0x" << static_cast<int>(id.sdid) << std::dec;
         return os;
     }
     /**
      * @brief: Input stream operator to parse identifier from string.
      *
      * Format: "0xDID:0xSDID" (e.g. "0x60:0x60")
      *
      * @param [in] is: The input stream.
      * @param [out] id: The ancillary data identifier.
      *
      * @return: The input stream.
      */
     friend std::istream& operator>>(std::istream& is, AncillaryDataIdentifier& id)
     {
         std::string token;
         is >> token;
         size_t colon_pos = token.find(':');
         if (colon_pos != std::string::npos) {
             id.did = static_cast<uint8_t>(std::stoul(token.substr(0, colon_pos), nullptr, 0));
             id.sdid = static_cast<uint8_t>(std::stoul(token.substr(colon_pos + 1), nullptr, 0));
         }
         return is;
     }
 };

/* Time constants */
constexpr size_t NS_IN_SEC = std::chrono::nanoseconds{ std::chrono::seconds{ 1 } }.count();
constexpr size_t NS_IN_USEC = std::chrono::nanoseconds{ std::chrono::microseconds{ 1 } }.count();
constexpr size_t NS_IN_MSEC = std::chrono::nanoseconds{ std::chrono::milliseconds{ 1 } }.count();
constexpr size_t USEC_IN_SEC = std::chrono::microseconds{ std::chrono::seconds{ 1 } }.count();
constexpr uint8_t LEAP_SECONDS = 37;
constexpr uint64_t DEFAULT_STREAM_START_OFFSET_NS = NS_IN_SEC;
/* Bit masks and sizes */
constexpr size_t BYTE_SIZE_BITS = 8;
constexpr size_t WORD_SIZE_BITS = 32;
constexpr uint32_t MASK_7BIT = 0x7F;
constexpr uint32_t MASK_8BIT = 0xFF;
constexpr uint32_t MASK_9BIT = 0x1FF;
constexpr uint32_t MASK_10BIT = 0x3FF;
constexpr uint32_t MASK_11BIT = 0x7FF;
constexpr uint32_t MASK_12BIT = 0xFFF;
/* Resolution constants */
constexpr size_t _1080_WIDTH = 1920;
constexpr size_t _1080_HEIGHT = 1080;
constexpr size_t _2160_WIDTH = 3840;
constexpr size_t _2160_HEIGHT = 2160;
constexpr size_t _4320_WIDTH = 7680;
constexpr size_t _4320_HEIGHT = 4320;
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
constexpr size_t RTP_ST_2110_40_HEADER_EXT_SIZE = 8;
constexpr size_t RTP_ST_2110_40_ANCILLARY_HEADER_SIZE = \
    RTP_HEADER_SIZE + RTP_ST_2110_40_HEADER_EXT_SIZE;
constexpr size_t RTP_ST_2110_40_FIXED_DATA_HEADER_SIZE = 4;
constexpr size_t RTP_ST_2110_40_DID_WORD_COUNT = 1;
constexpr size_t RTP_ST_2110_40_SDID_WORD_COUNT = 1;
constexpr size_t RTP_ST_2110_40_DATA_COUNT_WORD_COUNT = 1;
constexpr size_t RTP_ST_2110_40_DATA_WORD_SIZE_BITS = 10;
constexpr size_t RTP_ST_2110_40_CHECKSUM_SIZE_BITS = 10;
constexpr size_t RTP_ST_2110_40_PARITY_BIT_POSITION = 8;
constexpr size_t RTP_ST_2110_40_INVERSE_PARITY_BIT_POSITION = 9;
constexpr size_t RTP_ST_2110_40_C_FLAG_BIT_POSITION = 31;
constexpr size_t RTP_ST_2110_40_LINE_NUMBER_BIT_POSITION = 20;
constexpr size_t RTP_ST_2110_40_HORIZONTAL_OFFSET_BIT_POSITION = 8;
constexpr size_t RTP_ST_2110_40_S_FLAG_BIT_POSITION = 7;
constexpr size_t RTP_ST_2110_40_CHECKSUM_MSB_POSITION = 8;
constexpr size_t RTP_ST_2110_40_CHECKSUM_INVERSE_BIT_POSITION = 9;
constexpr uint32_t RTP_SEQUENCE_NUMBER_MASK_16BIT = 0xFFFF;
constexpr uint32_t RTP_SEQUENCE_NUMBER_MASK_32BIT = 0xFFFFFFFF;
constexpr uint8_t RTP_VERSION_MASK = 0xC0;
constexpr uint8_t RTP_VERSION_2 = 0x80;
constexpr uint8_t RTP_M_BIT_MASK = 0x80;
/* Video constants */
constexpr size_t VIDEO_TRO_DEFAULT_MODIFICATION = 2;
/* Ancillary data constants */
constexpr size_t DEFAULT_ANCILLARY_DATA_PACKETS_PER_PACKET = 10;
constexpr size_t DEFAULT_ANCILLARY_DATA_WORDS_COUNT = 128;
constexpr AncillaryDataIdentifier ANCILLARY_TIMECODE_IDENTIFIER = {0x60, 0x60};
constexpr AncillaryDataIdentifier ANCILLARY_AFD_IDENTIFIER = {0x41, 0x05};
constexpr AncillaryDataIdentifier ANCILLARY_CLOSED_CAPTION_IDENTIFIER = {0x61, 0x01};
/* Supported video resolutions */
const std::vector<Resolution> SUPPORTED_VIDEO_RESOLUTIONS = {
    { _1080_WIDTH, _1080_HEIGHT },
    { _2160_WIDTH, _2160_HEIGHT },
    { _4320_WIDTH, _4320_HEIGHT },
    { _1080_HEIGHT, _1080_WIDTH },
    { _2160_HEIGHT, _2160_WIDTH },
    { _4320_HEIGHT, _4320_WIDTH }
};
/* Supported video frame rates */
const std::vector<FrameRate> SUPPORTED_VIDEO_FRAME_RATES = {
    { 24000, 1001 },  /**< 23.976 fps */
    { 24 },           /**< 24 fps */
    { 25 },           /**< 25 fps */
    { 30000, 1001 },  /**< 29.97 fps */
    { 30 },           /**< 30 fps */
    { 50 },           /**< 50 fps */
    { 60000, 1001 },  /**< 59.94 fps */
    { 60 }            /**< 60 fps */
};
/* Supported video sampling types */
const std::vector<VideoSampling> SUPPORTED_VIDEO_SAMPLING_TYPES = {
    VideoSampling::YCbCr_4_4_4,
    VideoSampling::YCbCr_4_2_2,
    VideoSampling::YCbCr_4_2_0,
    VideoSampling::RGB
};
/* Supported video scan types */
const std::vector<VideoScanType> SUPPORTED_VIDEO_SCAN_TYPES = {
    VideoScanType::Progressive,
    VideoScanType::Interlaced
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

}  // namespace services
}  // namespace dev_kit
}  // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_MEDIA_DEFS_H_ */
