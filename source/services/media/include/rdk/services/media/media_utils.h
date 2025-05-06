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

#ifndef RDK_SERVICES_MEDIA_MEDIA_UTILS_H_
#define RDK_SERVICES_MEDIA_MEDIA_UTILS_H_

#include <cstddef>
#include <chrono>
#include <vector>
#include <unordered_map>

#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/sdp/sdp.h"
#include "rdk/services/settings/app_settings.h"
#include "rdk/services/media/media_defs.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 * @brief: Initializes media settings.
 *
 * This method is responsible to initialize media settings based on the
 * application settings. Currently, it's specific to SMPTE 2110-20.
 *
 * @param [in] settings: The application settings.
 * @param [in] extra_format_specific_parameters: The extra format specific parameters (default: empty).
 *
 * @return: The status of the operation.
 */
ReturnStatus initialize_media_settings(
    AppSettings& settings, const std::vector<FormatSpecificParameter>& extra_format_specific_parameters = {});
/**
 * @brief: Calculates TRO and TRS based on SMPTE 2110-21 standard.
 *
 * This method is responsible to calculate TRO and TRS based on the SMPTE 2110-21 standard.
 *
 * @param [in] media_settings: The media settings.
 * @param [out] tro: The calculated TRO.
 * @param [out] trs: The calculated TRS.
 */
void calculate_tro_trs(MediaSettings& media_settings, double& tro, double& trs);

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_MEDIA_UTILS_H_ */
