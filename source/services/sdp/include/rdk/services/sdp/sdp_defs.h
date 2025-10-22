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

#ifndef RDK_SERVICES_SDP_SDP_DEFS_H_
#define RDK_SERVICES_SDP_SDP_DEFS_H_

#include <cstdint>
#include <cstddef>

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Default RTP clock rates.
 *
 * This corresponds to <clock rate> field in a=rtpmap
 * as defined in RFC4566 section 6 and its value as defined
 * in RFC 1890 section 6.
 */
constexpr size_t VIDEO_RTP_CLOCK_RATE = 90000;
constexpr size_t ANCILLARY_RTP_CLOCK_RATE = 90000;
/**
 * @brief: Default RTP payload type for SMPTE ST 2110.
 *
 * This corresponds to <payload_type> filed in a=rtpmap
 * as defined in RFC4566 section 6. Values correspond
 * to the common SMPTE payload types.
 */
constexpr std::uint8_t PAYLOAD_TYPE_ST_2110_20 = 96;
constexpr std::uint8_t PAYLOAD_TYPE_ST_2110_30 = 97;
constexpr std::uint8_t PAYLOAD_TYPE_ST_2110_40 = 100;
/**
 * @brief: RTP payload type.
 *
 * This corresponds to RTP payload type which is 7-bit field
 * defined in RFC3550 section 5.1.
 */
constexpr std::uint8_t RTP_PAYLOAD_TYPE_MAX = 127;
/**
 * @brief: Default media format for SMPTE ST 2110.
 *
 * This corresponds to <media_format> filed in "a=fmtp" defined in ST 2110-20
 * and RFC 4566.
 */
constexpr std::uint8_t MEDIA_FORMAT_ST_2110_20 = PAYLOAD_TYPE_ST_2110_20;
constexpr std::uint8_t MEDIA_FORMAT_ST_2110_30 = PAYLOAD_TYPE_ST_2110_30;
constexpr std::uint8_t MEDIA_FORMAT_ST_2110_40 = PAYLOAD_TYPE_ST_2110_40;
/**
 * @brief: Enum class for network type.
 *
 * This corresponds to the <nettype> field in SDP as per RFC4566.
 */
enum class NetworkType
{
    _IN,

    Unknown
};
/**
 * @brief: Enum class for address type.
 *
 * This corresponds to the <addrtype> field in SDP as per RFC4566.
 */
enum class AddressType
{
    IP4,
    IP6,

    Unknown
};
/**
 * @brief: Enum class for network filter mode.
 *
 * This corresponds to the <filter-mode> field in "a=source-filter" attribute in SDP as per RFC4570.
 */
enum class NetworkFilterMode
{
    Inclusive,
    Exclusive,

    Unknown
};
/**
 * @brief: Enum class for media type.
 *
 * This corresponds to the <media> field in SDP as per RFC4566.
 */
enum class MediaType
{
    Video,
    Audio,

    Unknown
};
/**
 * @brief: Enum class for video sampling types.
 *
 * This corresponds to the <sampling> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-20.
 */
enum class VideoSampling
{
    YCbCr_4_4_4,
    YCbCr_4_2_2,
    YCbCr_4_2_0,
    CLYCbCr_4_4_4,
    CLYCbCr_4_2_2,
    CLYCbCr_4_2_0,
    ICtCp_4_4_4,
    ICtCp_4_2_2,
    ICtCp_4_2_0,
    RGB,
    XYZ,
    KEY,

    Unknown
};
/**
 * @brief: Enum class for video bit depth.
 *
 * This corresponds to the <depth> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-20.
 * This attribute is applicable to the main video essence as well as to key/alpha channel.
 */
enum class VideoBitDepth
{
    _8,
    _10,
    _12,
    _16,
    _16f,

    Unknown
};
/**
 * @brief: Enum class for colorimetry values.
 *
 * This corresponds to the <colorimetry> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-20.
 */
enum class Colorimetry
{
    BT601,
    BT709,
    BT2020,
    BT2100,
    ST2065_1,
    ST2065_3,
    UNSPECIFIED,
    XYZ,
    ALPHA,

    Unknown
};
/**
 * @brief: Enum class for packaging modes.
 *
 * This corresponds to the <PM> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-20.
 */
enum class PackagingMode
{
    _2110GPM,
    _2110BPM,

    Unknown
};
/**
 * @brief: Enum class for SMPTE standard numbers.
 *
 * This corresponds to the <SSN> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-20.
 */
enum class SMPTEStandardNumber
{
    ST2110_20_2017,
    ST2110_20_2021,

    Unknown
};
/**
 * @brief: Enum class for sender types.
 *
 * This corresponds to the <TP> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-21.
 */
enum class SenderType
{
    _2110TPN,
    _2110TPNL,
    _2110TPW,

    Unknown
};
/**
 * @brief: Enum class for video scan types.
 *
 * This corresponds to the <interlace> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-20.
 */
enum class VideoScanType
{
    Progressive,
    Interlaced,

    Unknown
};
/**
 * @brief: Enum class for timestamp modes.
 *
 * This corresponds to the <TSMODE> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-10.
 */
enum class TimestampMode
{
    SAMP,
    NEW,
    PRES,

    Unknown
};
/**
 * @brief: Enum class for timestamp reference clocks.
 *
 * This corresponds to the <ts-refclk> field in "a=ts-refclk" attribute in SDP as per SMPTE ST 2110-10.
 */
enum class TimestampRefClock
{
    PTP,
    LocalMAC,

    Unknown
};
/**
 * @brief: Enum class for media clocks.
 *
 * This corresponds to the <mediaclk> field in "a=mediaclk" attribute in SDP as per SMPTE ST 2110-10.
 */
enum class MediaClock
{
    Direct,
    Sender,

    Unknown
};
/**
 * @brief: Enum class for transport protocol.
 *
 * This corresponds to the <proto> field in "m=" attribute in SDP as per RFC4566.
 */
enum class TransportProtocol
{
    RTP_AVP,

    Unknown
};
/**
 * @brief: Enum class for audio sampling rates.
 *
 * This corresponds to the clock rate in "a=rtpmap" attribute in SDP as per SMPTE ST 2110-30.
 */
enum class AudioSamplingRate
{
    _44100,
    _48000,
    _96000,

    Unknown
};
/**
 * @brief: Enum class for audio encoding names.
 *
 * This corresponds to the encoding name in "a=rtpmap" attribute in SDP as per SMPTE ST 2110-30.
 */
enum class AudioEncoding
{
    L16,
    L20,
    L24,

    Unknown
};
/**
 * @brief: Enum class for audio channel grouping symbols for channel-order.
 *
 * This corresponds to the channel-order=<convention> field of the a=fmtp
 * as defined in SMPTE ST 2110-30:2017 section 6.2.2.
 * Short and long names map to the same underlying value.
 */
enum class ChannelGroupingSymbol
{
    M = 0,                 Mono = 0,
    DM = 1,                Dual_Mono = 1,
    ST = 2,                Standard_Stereo = 2,
    LtRt = 3,              Matrix_Stereo = 3,
    _51 = 4,               _5_1_Surround = 4,
    _71 = 5,               _7_1_Surround = 5,
    _222 = 6,              _22_2_Surround = 6,
    SGRP = 7,              One_SDI_Audio_Group = 7,

    U01 = 101,             Undefined01 = U01,
    U02,                   Undefined02 = U02,
    U03,                   Undefined03 = U03,
    U04,                   Undefined04 = U04,
    U05,                   Undefined05 = U05,
    U06,                   Undefined06 = U06,
    U07,                   Undefined07 = U07,
    U08,                   Undefined08 = U08,
    U09,                   Undefined09 = U09,
    U10,                   Undefined10 = U10,
    U11,                   Undefined11 = U11,
    U12,                   Undefined12 = U12,
    U13,                   Undefined13 = U13,
    U14,                   Undefined14 = U14,
    U15,                   Undefined15 = U15,
    U16,                   Undefined16 = U16,
    U17,                   Undefined17 = U17,
    U18,                   Undefined18 = U18,
    U19,                   Undefined19 = U19,
    U20,                   Undefined20 = U20,
    U21,                   Undefined21 = U21,
    U22,                   Undefined22 = U22,
    U23,                   Undefined23 = U23,
    U24,                   Undefined24 = U24,
    U25,                   Undefined25 = U25,
    U26,                   Undefined26 = U26,
    U27,                   Undefined27 = U27,
    U28,                   Undefined28 = U28,
    U29,                   Undefined29 = U29,
    U30,                   Undefined30 = U30,
    U31,                   Undefined31 = U31,
    U32,                   Undefined32 = U32,
    U33,                   Undefined33 = U33,
    U34,                   Undefined34 = U34,
    U35,                   Undefined35 = U35,
    U36,                   Undefined36 = U36,
    U37,                   Undefined37 = U37,
    U38,                   Undefined38 = U38,
    U39,                   Undefined39 = U39,
    U40,                   Undefined40 = U40,
    U41,                   Undefined41 = U41,
    U42,                   Undefined42 = U42,
    U43,                   Undefined43 = U43,
    U44,                   Undefined44 = U44,
    U45,                   Undefined45 = U45,
    U46,                   Undefined46 = U46,
    U47,                   Undefined47 = U47,
    U48,                   Undefined48 = U48,
    U49,                   Undefined49 = U49,
    U50,                   Undefined50 = U50,
    U51,                   Undefined51 = U51,
    U52,                   Undefined52 = U52,
    U53,                   Undefined53 = U53,
    U54,                   Undefined54 = U54,
    U55,                   Undefined55 = U55,
    U56,                   Undefined56 = U56,
    U57,                   Undefined57 = U57,
    U58,                   Undefined58 = U58,
    U59,                   Undefined59 = U59,
    U60,                   Undefined60 = U60,
    U61,                   Undefined61 = U61,
    U62,                   Undefined62 = U62,
    U63,                   Undefined63 = U63,
    U64,                   Undefined64 = U64,

    Unknown
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_SDP_SDP_DEFS_H_ */
