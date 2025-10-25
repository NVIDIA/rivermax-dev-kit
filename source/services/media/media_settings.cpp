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

#include "rdk/services/media/media_settings.h"
#include "rdk/services/media/media_settings_calculator.h"

using namespace rivermax::dev_kit::services;

void MediaSettings::set_calculator(std::shared_ptr<IMediaSettingsCalculator> calculator)
{
    media_settings_calculator = std::move(calculator);
}

ReturnStatus MediaSettings::create_default_calculator(const std::vector<FormatSpecificParameter>& extra_parameters)
{
    media_settings_calculator = IMediaSettingsCalculatorFactory::get_media_settings_calculator(*this, extra_parameters);
    if (!media_settings_calculator) {
        return ReturnStatus::failure;
    }
    return media_settings_calculator->calculate_media_settings();
}
