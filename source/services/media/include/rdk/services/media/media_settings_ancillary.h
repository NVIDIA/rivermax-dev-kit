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

#ifndef RDK_SERVICES_MEDIA_MEDIA_SETTINGS_ANCILLARY_H_
#define RDK_SERVICES_MEDIA_MEDIA_SETTINGS_ANCILLARY_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "rdk/services/media/media_calc_interface.h"
#include "rdk/services/media/media_settings.h"
#include "rdk/services/settings/app_settings.h"

namespace rdk
{
namespace services
{

/**
 * @brief: SMPTE 2110-40 ancillary media settings.
 *
 * The struct will be used to hold media parameters for
 * SMPTE 2110-40 ancillary data streams.
 */
struct SMPTE_2110_40_MediaSettings : public MediaSettings
{
    /**
     * @brief: Default constructor.
     */
    SMPTE_2110_40_MediaSettings() = default;
    /**
     * @brief: Constructor from AppSettings.
     *
     * @param [in] app_settings: Application settings to configure from.
     */
    explicit SMPTE_2110_40_MediaSettings(const AppSettings& app_settings);
    virtual ~SMPTE_2110_40_MediaSettings() = default;
    virtual SMPTEStandard get_smpte_standard() const override { return SMPTEStandard::ST_2110_40; }
    virtual bool needs_dynamic_packet_sizes() const override { return true; }

    std::vector<AncillaryDataIdentifier> data_identifiers;
    uint16_t max_user_data_words_count = DEFAULT_ANCILLARY_DATA_WORDS_COUNT;
    uint16_t max_ancillary_data_packets_per_packet = DEFAULT_ANCILLARY_DATA_PACKETS_PER_PACKET;
    FrameRate frame_rate = {60};
    VideoScanType video_scan_type = VideoScanType::Progressive;
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_MEDIA_MEDIA_SETTINGS_ANCILLARY_H_ */
