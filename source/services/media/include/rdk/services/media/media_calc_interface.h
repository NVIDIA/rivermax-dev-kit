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

#ifndef RDK_SERVICES_MEDIA_MEDIA_CALC_INTERFACE_H_
#define RDK_SERVICES_MEDIA_MEDIA_CALC_INTERFACE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/media/media_defs.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

struct MediaSettings;

/**
 * @brief: Simple network flow information for SDP generation.
 */
struct NetworkFlow
{
    std::string source_ip;
    std::string destination_ip;
    uint16_t destination_port;

    NetworkFlow(const std::string& src_ip, const std::string& dst_ip, uint16_t dst_port)
        : source_ip(src_ip), destination_ip(dst_ip), destination_port(dst_port) {}
};

/**
 * @brief: Interface for media settings calculators.
 *
 * This abstract class defines the interface that all media settings calculators
 * must implement. It provides methods for calculating media settings and composing
 * SDP descriptions for different media types according to SMPTE standards.
 */
class IMediaSettingsCalculator
{
public:
    /**
     * @brief: Virtual destructor.
     */
    virtual ~IMediaSettingsCalculator() = default;
    /**
     * @brief: Calculates media settings for the specific media type.
     *
     * This pure virtual method must be implemented by derived classes to perform
     * all necessary calculations to configure the media settings for their specific
     * media type, including timing, packetization, and memory layout parameters.
     *
     * @return: Status of the calculation operation.
     */
    virtual ReturnStatus calculate_media_settings() = 0;
    /**
     * @brief: Generates SDP description for the media type.
     *
     * This pure virtual method must be implemented by derived classes to generate
     * a Session Description Protocol (SDP) description string for their specific
     * media type with the specified network parameters. For a single flow, generates
     * a standard SDP. For multiple flows, generates an SDP with SMPTE 2022-7 duplication
     * including group description and media IDs (a, b, c, etc.).
     *
     * @param [in] flows: Vector of flow info containing source IP, destination IP and port.
     *
     * @return: SDP description string for the media stream.
     */
    virtual std::string generate_media_sdp(const std::vector<NetworkFlow>& flows) = 0;
    /**
     * @brief: Returns the SMPTE standard name.
     *
     * This pure virtual method must be implemented by derived classes to return
     * a string representation of their specific SMPTE standard name.
     *
     * @return: String representation of the SMPTE standard.
     */
    virtual std::string get_smpte_standard_name() const = 0;
    /**
     * @brief: Aligns a desired time to the nearest media unit boundary for the specific media type.
     *
     * This pure virtual method must be implemented by derived classes to align
     * a given timestamp to the next valid media unit boundary based on their specific
     * media type timing requirements and SMPTE standard specifications.
     *
     * @param [in] desired_time_ns: Desired time in nanoseconds to align.
     *
     * @return: Aligned time in nanoseconds at the media unit boundary.
     */
    virtual double align_time_to_media_unit_boundary_ns(uint64_t desired_time_ns) const = 0;
    /**
     * @brief: Returns the transmit offset value in nanoseconds.
     *
     * This method returns the transmit offset value for the media type,
     * which is the time offset between the frame boundary and when packets should
     * actually be transmitted.
     *
     * @return: Transmit offset value in nanoseconds.
     */
    virtual double get_transmit_offset_ns() const { return 0.0; }
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_MEDIA_CALC_INTERFACE_H_ */
