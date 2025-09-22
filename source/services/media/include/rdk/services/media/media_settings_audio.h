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

#ifndef RDK_SERVICES_MEDIA_MEDIA_SETTINGS_AUDIO_H_
#define RDK_SERVICES_MEDIA_MEDIA_SETTINGS_AUDIO_H_

#include <cstdint>
#include <unordered_set>
#include <cstddef>

#include "rdk/services/media/media_settings.h"


namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Audio sampling frequencies for SMPTE ST 2110-30.
 *
 * This corresponds to the <sampling-frequency> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-30.
 */
enum class AudioSamplingFrequency : uint32_t
{
    _48000 = 48000,
    _96000 = 96000,
    _192000 = 192000,

    Unknown = 0
};

/**
 * @brief: Audio bit depths for SMPTE ST 2110-30.
 *
 * This corresponds to the <encoding> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-30.
 */
enum class AudioBitDepth : uint8_t
{
    _16 = 16,
    _24 = 24,
    _32 = 32,

    Unknown = 0
};


/* Supported audio sampling frequencies */
const std::unordered_set<AudioSamplingFrequency> SUPPORTED_AUDIO_SAMPLING_FREQUENCIES = {
    AudioSamplingFrequency::_48000,
    AudioSamplingFrequency::_96000,
    AudioSamplingFrequency::_192000
};

/* Supported audio bit depths */
const std::unordered_set<AudioBitDepth> SUPPORTED_AUDIO_BIT_DEPTHS = {
    AudioBitDepth::_16,
    AudioBitDepth::_24,
    AudioBitDepth::_32
};

/* Supported audio channel counts */
const std::unordered_set<uint8_t> SUPPORTED_AUDIO_CHANNEL_COUNTS = {
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
    virtual ~SMPTE_2110_30_MediaSettings() = default;
    virtual SMPTEStandard get_media_type() const override { return SMPTEStandard::ST_2110_30; }

    AudioSamplingFrequency sampling_frequency = AudioSamplingFrequency::_48000;
    AudioBitDepth bit_depth = AudioBitDepth::_24;
    uint8_t num_channels = 2;
    uint32_t ptime_usec = 1000;
    
    // Calculated parameters
    uint32_t samples_per_packet = 0;
    size_t bytes_per_sample = 0;
    uint32_t packets_per_second = 0;
};

}  // namespace services
}  // namespace dev_kit
}  // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_MEDIA_SETTINGS_AUDIO_H_ */
