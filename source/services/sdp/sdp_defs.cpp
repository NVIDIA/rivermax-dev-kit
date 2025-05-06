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

#include "rdk/services/utils/enum_utils.h"
#include "rdk/services/sdp/sdp_defs.h"

using namespace rivermax::dev_kit::services;

template <>
constexpr EnumNameArray<NetworkType> EnumMapper<NetworkType>::names = {
    "IN"
};
template <>
constexpr EnumNameArray<AddressType> EnumMapper<AddressType>::names = {
    "4", "6"
};
template <>
constexpr EnumNameArray<NetworkFilterMode> EnumMapper<NetworkFilterMode>::names = {
    "incl", "excl"
};
template <>
constexpr EnumNameArray<MediaType> EnumMapper<MediaType>::names = {
    "video", "audio"
};
template <>
constexpr EnumNameArray<VideoSampling> EnumMapper<VideoSampling>::names = {
    "YCbCr-4:4:4", "YCbCr-4:2:2", "YCbCr-4:2:0", "CLYCbCr-4:4:4", "CLYCbCr-4:2:2", "CLYCbCr-4:2:0",
    "ICtCp-4:4:4", "ICtCp-4:2:2", "ICtCp-4:2:0", "RGB", "XYZ", "KEY"
};
template <>
constexpr EnumNameArray<ColorBitDepth> EnumMapper<ColorBitDepth>::names = {
    "8", "10", "12", "16", "16f"
};
template <>
constexpr EnumNameArray<Colorimetry> EnumMapper<Colorimetry>::names = {
    "BT601", "BT709", "BT2020", "BT2100", "ST2065-1", "ST2065-3", "UNSPECIFIED", "XYZ", "ALPHA"
};
template <>
constexpr EnumNameArray<PackagingMode> EnumMapper<PackagingMode>::names = {
    "2110GPM", "2110BPM"
};
template <>
constexpr EnumNameArray<SMPTEStandardNumber> EnumMapper<SMPTEStandardNumber>::names = {
    "ST2110-20:2017", "ST2110-20:2021"
};
template <>
constexpr EnumNameArray<SenderType> EnumMapper<SenderType>::names = {
    "2110TPN", "2110TPNL", "2110TPW"
};
template <>
constexpr EnumNameArray<VideoScanType> EnumMapper<VideoScanType>::names = {
    "Progressive", "Interlaced"
};
template <>
constexpr EnumNameArray<TimestampMode> EnumMapper<TimestampMode>::names = {
    "SAMP", "NEW", "PRES"
};
template <>
constexpr EnumNameArray<TimestampRefClock> EnumMapper<TimestampRefClock>::names = {
    "ptp", "localmac"
};
template <>
constexpr EnumNameArray<MediaClock> EnumMapper<MediaClock>::names = {
    "direct", "sender"
};
template <>
constexpr EnumNameArray<TransportProtocol> EnumMapper<TransportProtocol>::names = {
    "RTP/AVP"
};
