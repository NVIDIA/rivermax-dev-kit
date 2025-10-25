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

#include <cstdint>
#include <unordered_map>

#include "rt_threads.h"

#include "rdk/services/media/media_settings_calculator.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/media/ancillary_settings_calculator.h"
#include "rdk/services/media/audio_settings_calculator.h"
#include "rdk/services/media/media_calc_interface.h"
#include "rdk/services/media/media_settings.h"
#include "rdk/services/media/video_settings_calculator.h"
#include "rdk/services/sdp/sdp.h"
#include "rdk/services/sdp/sdp_defs.h"

using namespace rivermax::dev_kit::services;

media_settings_calculator_factory_map_t IMediaSettingsCalculatorFactory::s_media_settings_calculator_factory_map = {
    {
        SMPTEStandard::ST_2110_20,
        [](MediaSettings& media_settings, const std::vector<FormatSpecificParameter>& extra_parameters)
        {
            // Runtime type safety check
            if (media_settings.get_smpte_standard() != SMPTEStandard::ST_2110_20) {
                throw std::invalid_argument("MediaSettings type mismatch: expected ST_2110_20, got " +
                                          std::to_string(static_cast<int>(media_settings.get_smpte_standard())));
            }
            auto& video_settings = static_cast<SMPTE_2110_20_MediaSettings&>(media_settings);
            return std::make_shared<ST_2110_20_MediaSettingsCalculator>(video_settings, extra_parameters);
        }
    },
    {
        SMPTEStandard::ST_2110_30,
        [](MediaSettings& media_settings, const std::vector<FormatSpecificParameter>& extra_parameters)
        {
            // Runtime type safety check
            if (media_settings.get_smpte_standard() != SMPTEStandard::ST_2110_30) {
                throw std::invalid_argument("MediaSettings type mismatch: expected ST_2110_30, got " +
                                          std::to_string(static_cast<int>(media_settings.get_smpte_standard())));
            }
            auto& audio_settings = static_cast<SMPTE_2110_30_MediaSettings&>(media_settings);
            return std::make_shared<ST_2110_30_MediaSettingsCalculator>(audio_settings, extra_parameters);
        }
    },
    {
        SMPTEStandard::ST_2110_40,
        [](MediaSettings& media_settings, const std::vector<FormatSpecificParameter>& extra_parameters)
        {
            // Runtime type safety check
            if (media_settings.get_smpte_standard() != SMPTEStandard::ST_2110_40) {
                throw std::invalid_argument("MediaSettings type mismatch: expected ST_2110_40, got " +
                                          std::to_string(static_cast<int>(media_settings.get_smpte_standard())));
            }
            auto& ancillary_settings = static_cast<SMPTE_2110_40_MediaSettings&>(media_settings);
            return std::make_shared<ST_2110_40_MediaSettingsCalculator>(ancillary_settings, extra_parameters);
        }
    }
};

std::shared_ptr<IMediaSettingsCalculator> IMediaSettingsCalculatorFactory::get_media_settings_calculator(MediaSettings& media_settings,
    const std::vector<FormatSpecificParameter>& extra_parameters)
{
    auto iter = s_media_settings_calculator_factory_map.find(media_settings.get_smpte_standard());
    if (iter != s_media_settings_calculator_factory_map.end()) {
        return iter->second(media_settings, extra_parameters);
    }
    return nullptr;
}
