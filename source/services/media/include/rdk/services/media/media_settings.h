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

#ifndef RDK_SERVICES_MEDIA_MEDIA_SETTINGS_H_
#define RDK_SERVICES_MEDIA_MEDIA_SETTINGS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rdk/services/media/media_calc_interface.h"
#include "rdk/services/media/media_defs.h"
#include "rdk/services/sdp/sdp_common_descriptions.h"
#include "rdk/services/sdp/sdp_defs.h"
#include "rdk/services/utils/rational.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 * @brief: Media settings.
 *
 * The struct will be used to hold media parameters for
 * all different media types processed by the application.
 */
struct MediaSettings
{
    virtual ~MediaSettings() = default;
    /**
     * @brief: Returns the SMPTE standard.
     *
     * @return: SMPTE standard.
     */
    virtual SMPTEStandard get_smpte_standard() const = 0;
    /**
     * @brief: Returns if dynamic packet sizes are needed.
     *
     * @return: True if dynamic packet sizes are needed, false otherwise.
     */
     virtual bool needs_dynamic_packet_sizes() const { return false; }
    /**
     * @brief: Returns packets per frame as expected by Rivermax API.
     *
     * @return: Number of packets per frame.
     */
    virtual size_t get_packets_per_frame() const { return packets_in_media_unit; }
    /**
     * @brief: Set the media settings calculator.
     *
     * @param [in] calculator: Shared pointer to the calculator.
     */
    void set_calculator(std::shared_ptr<IMediaSettingsCalculator> calculator);
    /**
     * @brief: Create the default calculator.
     *
     * @param [in] extra_parameters: Optional format-specific parameters for calculator creation.
     *
     * @return: Status of the creation operation.
     */
    ReturnStatus create_default_calculator(const std::vector<FormatSpecificParameter>& extra_parameters = {});

    /* Constants */
    static constexpr size_t MIN_MEDIA_UNITS_FOR_SIMULTANEOUS_TX_AND_FILLUP = 2;
    static constexpr uint32_t MAX_PAYLOAD_SIZE = 1440;
    static constexpr size_t DEFAULT_NUM_OF_MEDIA_UNITS_IN_MEM_BLOCK = 10;
    static constexpr size_t DEFAULT_NUM_OF_MEM_BLOCKS = 1;
    static constexpr uint8_t DEFAULT_PTP_DOMAIN_ID = 127;
    static constexpr uint8_t DEFAULT_PAYLOAD_TYPE = 96;
    static constexpr size_t RTP_SAMPLE_RATE = 90000;
    uint32_t sdp_media_block_index = 0;
    uint8_t payload_type = DEFAULT_PAYLOAD_TYPE;
    std::string media_file = "";
    bool dynamic_media_file_load = false;
    std::shared_ptr<IMediaSettingsCalculator> media_settings_calculator;
    size_t sample_rate = RTP_SAMPLE_RATE;
    double media_unit_time_interval_ns = 0;
    Rational ticks_per_media_unit;
    /**
     * @brief: Reference clock ID.
     *
     * The reference clock ID is used to identify the reference clock.
     * If the reference clock is PTP, the ID is either the PTP grandmaster clock identity,
     * or the reference clock source is defined as traceable (if the ID is empty).
     * If the reference clock is not PTP, the ID is the local MAC address.
     */
    std::string refclk_id = "";
    /**
     * @brief: Reference clock is PTP.
     */
    bool ref_clk_is_ptp = true;
    uint8_t ptp_domain_id = DEFAULT_PTP_DOMAIN_ID;
    uint32_t packets_in_media_unit = 0;
    size_t chunks_in_media_unit = 0;
    size_t packets_in_chunk = 0;
    uint16_t protocol_header_size = 0;
    uint16_t raw_packet_payload_size = 0;
    uint16_t packet_payload_size = 0;
    bool header_data_split = false;
    uint16_t packet_app_header_size = 0;
    size_t bytes_per_media_unit = 0;
    size_t media_units_in_mem_block = 0;
    size_t chunks_in_mem_block = 0;
    size_t packets_in_mem_block = 0;
    size_t requested_num_of_mem_blocks = DEFAULT_NUM_OF_MEM_BLOCKS;
    size_t data_stride_size = 0;
    size_t app_header_stride_size = 0;
};

}  // namespace services
}  // namespace dev_kit
}  // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_MEDIA_SETTINGS_H_ */
