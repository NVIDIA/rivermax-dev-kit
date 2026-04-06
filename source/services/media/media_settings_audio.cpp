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

#include "rdk/services/media/media_settings_audio.h"

using namespace rdk::services;

SMPTE_2110_30_MediaSettings::SMPTE_2110_30_MediaSettings(const AppSettings& app_settings)
{
    // Initialize base MediaSettings common fields
    header_data_split = app_settings.header_data_split;
    requested_num_of_mem_blocks = MediaSettings::DEFAULT_NUM_OF_MEM_BLOCKS;
    media_units_in_mem_block = app_settings.media.frames_fields_in_mem_block;
    payload_type = PAYLOAD_TYPE_ST_2110_30;
    media_file = app_settings.audio_file;
    dynamic_media_file_load = app_settings.dynamic_media_file_load;

    // Initialize SMPTE 2110-30 specific fields
    sampling_rate = app_settings.media.audio_sampling_rate;
    encoding = app_settings.media.audio_encoding;
    num_channels = app_settings.media.audio_channels_num;
    ptime_usec = app_settings.media.ptime_us;
}
