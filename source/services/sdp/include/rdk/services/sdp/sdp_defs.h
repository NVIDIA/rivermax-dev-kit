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

namespace rivermax
{
namespace dev_kit
{
namespace services
{

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
 * @brief: Enum class for color bit depth.
 *
 * This corresponds to the <depth> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-20.
 */
enum class ColorBitDepth
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

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_SDP_SDP_DEFS_H_ */
