/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef RMAX_APPS_LIB_LIB_SERVICES_UTILS_RTP_VIDEO_H_
#define RMAX_APPS_LIB_LIB_SERVICES_UTILS_RTP_VIDEO_H_

#include <cstddef>
#include <iostream>
#include <ostream>
#include <chrono>

#include <rivermax_api.h>

#include "api/rmax_apps_lib_api.h"

namespace ral
{
namespace lib
{
namespace services
{

constexpr size_t NS_IN_SEC = std::chrono::nanoseconds{ std::chrono::seconds{ 1 } }.count();
constexpr uint8_t LEAP_SECONDS = 37;

constexpr uint16_t FHD_WIDTH = 1920;
constexpr uint16_t FHD_HEIGHT = 1080;
constexpr uint16_t UHD_HEIGHT = 2160;
constexpr uint16_t UHD_WIDTH = 3840;
constexpr uint8_t VIDEO_TRO_DEFAULT_MODIFICATION = 2;
constexpr size_t HD_PACKETS_PER_FRAME_422_10B = 4320;

constexpr const char* VIDEO_2110_20_1080p50 = "1080p50";
constexpr const char* VIDEO_2110_20_1080p60 = "1080p60";
constexpr const char* VIDEO_2110_20_2160p50 = "2160p50";
constexpr const char* VIDEO_2110_20_2160p60 = "2160p60";
const std::unordered_set<const char*> SUPPORTED_STREAMS = {
    VIDEO_2110_20_1080p50,
    VIDEO_2110_20_1080p60,
    VIDEO_2110_20_2160p50,
    VIDEO_2110_20_2160p60
};

/**
 * @brief: Compose @ref media_settings_t for the given media stream.
 *
 * This helper function calculates media settings for the selected IPMX video stream format.
 */
void compose_ipmx_media_settings(AppSettings& s);
/**
 * @brief: Compose @ref media_settings_t for the given media stream.
 *
 * This helper function calculates media settings for the selected video stream format.
 */
void compose_media_settings(AppSettings& s);
/**
 * @brief: Calculate timing parameters for the given media stream.
 *
 * This helper function calculates media settings for the selected video stream format.
 */
void calculate_tro_trs(media_settings_t& media_settings, double& tro, double& trs);

} // namespace services
} // namespace lib
} // namespace ral

#endif /* RMAX_APPS_LIB_LIB_SERVICES_UTILS_RTP_VIDEO_H_ */
