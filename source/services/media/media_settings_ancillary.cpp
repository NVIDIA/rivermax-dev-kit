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

#include "rdk/services/media/media_settings_ancillary.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

SMPTE_2110_40_MediaSettings::SMPTE_2110_40_MediaSettings(const AppSettings& app_settings)
{
    // Initialize base MediaSettings common fields
    header_data_split = app_settings.header_data_split;
    requested_num_of_mem_blocks = MediaSettings::DEFAULT_NUM_OF_MEM_BLOCKS;
    media_units_in_mem_block = app_settings.media.frames_fields_in_mem_block;
    media_file = "";
    dynamic_media_file_load = false;

    // Initialize SMPTE 2110-40 specific fields
    did = app_settings.media.anc_did;
    sdid = app_settings.media.anc_sdid;
    user_data_size_bytes = app_settings.media.anc_data_size;
    frame_rate = app_settings.media.frame_rate;
}

} // namespace services
} // namespace dev_kit
} // namespace rivermax
