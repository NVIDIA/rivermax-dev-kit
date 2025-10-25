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

#ifndef RDK_SERVICES_MEDIA_MEDIA_SETTINGS_CALCULATOR_H_
#define RDK_SERVICES_MEDIA_MEDIA_SETTINGS_CALCULATOR_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "rdk/services/media/media_calc_interface.h"
#include "rdk/services/media/media_settings.h"
#include "rdk/services/sdp/sdp.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 * @brief: Media settings calculator factory map type.
 *
 * This typedef defines a map that associates SMPTE standards with factory functions
 * that create media settings calculators for specific media types.
 */
typedef const std::unordered_map<
    SMPTEStandard,
    std::function<std::shared_ptr<IMediaSettingsCalculator>(
        MediaSettings &media_settings,
        const std::vector<FormatSpecificParameter> &extra_parameters)>>
    media_settings_calculator_factory_map_t;

/**
 * @brief: Factory interface for creating media settings calculators.
 *
 * This class provides a static factory method to create media settings calculators
 * based on the SMPTE standard and media settings.
 */
class IMediaSettingsCalculatorFactory
{
private:
    /* Static map of SMPTE standards to calculator factory functions */
    static media_settings_calculator_factory_map_t s_media_settings_calculator_factory_map;
public:
    /**
     * @brief: Returns a media-type-specific media settings calculator for the media setting.
     *
     * This static method creates and returns a media settings calculator based on
     * the SMPTE standard of media settings provided.
     *
     * @param [in] media_settings: Reference to the media settings to configure.
     * @param [in] extra_parameters: Optional vector of format-specific parameters.
     *
     * @return: Shared pointer to the created media settings calculator.
     */
    static std::shared_ptr<IMediaSettingsCalculator> get_media_settings_calculator(MediaSettings& media_settings,
        const std::vector<FormatSpecificParameter>& extra_parameters = {});
};

/**
 * @brief: Base implementation of media settings calculator.
 *
 * This template class provides type-safe access to specific media settings types.
 * Derived classes inherit from this template with their specific settings type
 * ensuring compile-time type safety.
 *
 * @tparam SettingsType: The specific media settings type (e.g., SMPTE_2110_20_MediaSettings).
 */
template<typename SettingsType>
class MediaSettingsCalculator : public IMediaSettingsCalculator
{
public:
    /**
     * @brief: MediaSettingsCalculator constructor.
     *
     * @param [in] media_settings: Reference to the typed media settings to configure.
     * @param [in] extra_parameters: Optional vector of format-specific SDP parameters.
     */
    MediaSettingsCalculator(SettingsType& media_settings, const std::vector<FormatSpecificParameter>& extra_parameters = {}) :
        m_media_settings(media_settings), m_extra_parameters(extra_parameters) {}
    /**
     * @brief: Virtual destructor.
     */
    virtual ~MediaSettingsCalculator() = default;
    /**
     * @brief: Returns the name of the SMPTE standard.
     *
     * @return: String representation of the SMPTE standard name.
     */
    virtual std::string get_smpte_standard_name() const override { return "Unknown"; }
protected:
    /* Reference to the typed media settings being configured */
    SettingsType& m_media_settings;
    /* Vector of format-specific parameters for the SMPTE standard */
    const std::vector<FormatSpecificParameter> m_extra_parameters;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_MEDIA_SETTINGS_CALCULATOR_H_ */
