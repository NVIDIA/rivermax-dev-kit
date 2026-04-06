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

#ifndef RDK_SERVICES_MEDIA_MEDIA_SETTINGS_AUDIO_H_
#define RDK_SERVICES_MEDIA_MEDIA_SETTINGS_AUDIO_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "rdk/services/media/media_calc_interface.h"
#include "rdk/services/media/media_settings.h"
#include "rdk/services/sdp/sdp_defs.h"
#include "rdk/services/settings/app_settings.h"

namespace rdk
{
namespace services
{
/* Supported audio sampling frequencies */
const std::vector<AudioSamplingRate> SUPPORTED_AUDIO_SAMPLING_RATES = {
    AudioSamplingRate::_44100,
    AudioSamplingRate::_48000,
    AudioSamplingRate::_96000,
};

/* Supported audio bit depths */
const std::vector<AudioEncoding> SUPPORTED_AUDIO_ENCODINGS = {
    AudioEncoding::L16,
    AudioEncoding::L24
};

/* Supported audio channel counts */
const std::vector<uint8_t> SUPPORTED_AUDIO_CHANNEL_COUNTS = {
    1,  // Mono
    2,  // Stereo
    6,  // 5.1 surround
    8   // 7.1 surround
};

/**
 * @brief: SMPTE 2110-30 audio media settings.
 *
 * The struct will be used to hold media parameters for
 * SMPTE 2110-30 audio streams according to AES67 and ST 2110-30 standards.
 */
struct SMPTE_2110_30_MediaSettings : public MediaSettings
{
    /**
     * @brief: Default constructor.
     */
    SMPTE_2110_30_MediaSettings() = default;
    /**
     * @brief: Constructor from AppSettings.
     *
     * @param [in] app_settings: Application settings to configure from.
     */
    explicit SMPTE_2110_30_MediaSettings(const AppSettings& app_settings);
    virtual ~SMPTE_2110_30_MediaSettings() = default;
    virtual SMPTEStandard get_smpte_standard() const override { return SMPTEStandard::ST_2110_30; }

    // ST 2110-30 specific parameters
    AudioSamplingRate sampling_rate = AudioSamplingRate::_48000;
    AudioEncoding encoding = AudioEncoding::L24;
    uint8_t num_channels = 2;
    uint32_t ptime_usec = 1000;
    // Calculated parameters
    uint32_t samples_per_packet = 0;
    size_t bytes_per_sample = 0;
    uint32_t packets_per_second = 0;
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_MEDIA_MEDIA_SETTINGS_AUDIO_H_ */
