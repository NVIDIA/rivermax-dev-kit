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

#include "rdk/services/media/media_settings_ancillary.h"

using namespace rivermax::dev_kit::services;

SMPTE_2110_40_MediaSettings::SMPTE_2110_40_MediaSettings(const AppSettings& app_settings)
{
    // Initialize base MediaSettings common fields
    header_data_split = app_settings.header_data_split;
    requested_num_of_mem_blocks = MediaSettings::DEFAULT_NUM_OF_MEM_BLOCKS;
    media_units_in_mem_block = app_settings.media.frames_fields_in_mem_block;
    payload_type = PAYLOAD_TYPE_ST_2110_40;
    media_file = app_settings.ancillary_file;
    dynamic_media_file_load = false;

    // Initialize SMPTE 2110-40 specific fields
    data_identifiers = app_settings.media.ancillary_data_identifiers;
    if (data_identifiers.empty()) {
        data_identifiers.push_back(ANCILLARY_TIMECODE_IDENTIFIER);
        data_identifiers.push_back(ANCILLARY_AFD_IDENTIFIER);
    }
    max_user_data_words_count = app_settings.media.ancillary_data_words_count;
    frame_rate = app_settings.media.frame_rate;
    video_scan_type = app_settings.media.video_scan_type;
}
