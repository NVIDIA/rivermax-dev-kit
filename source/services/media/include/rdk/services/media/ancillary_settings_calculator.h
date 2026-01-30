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

#ifndef RDK_SERVICES_MEDIA_ANCILLARY_SETTINGS_CALCULATOR_H_
#define RDK_SERVICES_MEDIA_ANCILLARY_SETTINGS_CALCULATOR_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/media/media_settings_ancillary.h"
#include "rdk/services/media/media_settings_calculator.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: SMPTE 2110-40 ancillary media settings calculator.
 *
 * This class implements the media settings calculator for SMPTE 2110-40 ancillary data streams.
 * It provides methods to calculate ancillary-specific media settings and compose SDP descriptions
 * according to the SMPTE 2110-40 standard.
 */
class ST_2110_40_MediaSettingsCalculator : public MediaSettingsCalculator<SMPTE_2110_40_MediaSettings>
{
public:
    /**
     * @brief: ST_2110_40_MediaSettingsCalculator constructor.
     *
     * @param [in] media_settings: Reference to the media settings to configure.
     * @param [in] extra_parameters: Optional vector of format-specific parameters.
     */
    ST_2110_40_MediaSettingsCalculator(SMPTE_2110_40_MediaSettings& media_settings, const std::vector<FormatSpecificParameter>& extra_parameters = {}) :
        MediaSettingsCalculator<SMPTE_2110_40_MediaSettings>(media_settings, extra_parameters) {}
    /**
     * @brief: Virtual destructor.
     */
    virtual ~ST_2110_40_MediaSettingsCalculator() = default;
    /**
     * @brief: Calculates media settings for SMPTE 2110-40 ancillary data.
     *
     * This method performs all necessary calculations to configure the media settings
     * for a SMPTE 2110-40 ancillary data stream, including timing, packetization, and memory layout
     * according to SMPTE 2110-40 standard.
     *
     * @return: Status of the calculation operation.
     */
    virtual ReturnStatus calculate_media_settings() override;
    /**
     * @brief: Generates SDP description for SMPTE 2110-40 ancillary data.
     *
     * This method generates a Session Description Protocol (SDP) description string
     * for the SMPTE 2110-40 ancillary data stream with the specified network parameters.
     * For multiple flows, includes group description and media IDs for SMPTE 2022-7 duplication.
     *
     * @param [in] flows: Vector of flow info containing source IP, destination IP and port.
     *
     * @return: SDP description string for the ancillary data stream.
     */
    virtual std::string generate_media_sdp(const std::vector<NetworkFlow>& flows) override;
    /**
     * @brief: Returns the SMPTE standard name.
     *
     * @return: String representation of the SMPTE 2110-40 ancillary data SMPTE standard.
     */
    virtual std::string get_smpte_standard_name() const override;
    virtual double align_time_to_media_unit_boundary_ns(uint64_t desired_time_ns) const override;
private:
    /**
     * @brief: Calculates packet-specific parameters.
     *
     * @return: Status of the calculation operation.
     */
    ReturnStatus calculate_packet_parameters();
    /**
     * @brief: Calculates timing parameters.
     */
    void calculate_timing_parameters();
    /**
     * @brief: Calculates memory layout parameters.
     */
    void calculate_memory_parameters();
    /**
     * @brief: Calculates stride parameters for memory alignment.
     */
    void calculate_stride_parameters();
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_ANCILLARY_SETTINGS_CALCULATOR_H_ */
