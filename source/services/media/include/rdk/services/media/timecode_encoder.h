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

#ifndef RDK_SERVICES_MEDIA_TIMECODE_ENCODER_H_
#define RDK_SERVICES_MEDIA_TIMECODE_ENCODER_H_

#include <cstdint>
#include <vector>

#include "rdk/services/media/ancillary_data_encoder_interface.h"

namespace rdk
{
namespace services
{

/**
 * @brief: Timecode payload types.
 */
enum class TimecodePayloadType : uint8_t
{
    LTC,
    VITC1,
    VITC2
};

/**
 * @brief: SMPTE 12M-2 Timecode encoder.
 *
 * Encodes timecode in S12M-2 format with supporting different payload types
 * LTC, VITC1 and VITC2.
 *
 * Encoded time will represent the start time plus the frame number for the given frame rate.
 * Frame number will be repeated for fps higher than 30 (00, 00, 01, 01, 02, 02, ...) since
 * S12M-2 timecode only supports fps up to 30.
 *
 * Drop frame timecode is not supported.
 *
 * It is based on the SMPTE ST 12 specification.
 */
class TimecodeEncoder : public IAncillaryDataEncoder
{
public:
    /**
     * @brief: Construct timecode encoder.
     *
     * @param [in] frame_rate: Frame rate.
     * @param [in] start_time_ns: Start time in nanoseconds.
     * @param [in] payload_types: Payload types to encode.
     */
    TimecodeEncoder(const FrameRate& frame_rate, uint64_t start_time_ns, const std::vector<TimecodePayloadType>& payload_types);
    size_t get_packet_count() const override;
    size_t write_data(size_t packet_index, uint8_t* buffer) override;
    void fill_descriptor_header(size_t packet_index, AncillaryDataHeader& header) const override;
    void reset() override;
    /**
     * @brief: Sets the current frame.
     *
     * @param [in] frame: Current frame.
     */
    void set_current_frame(uint32_t frame) { m_current_frame = frame; }
    /**
     * @brief: Sets the start time.
     *
     * @param [in] time_ns: Start time in nanoseconds.
     */
    void set_start_time(uint64_t time_ns) { m_start_time_ns = time_ns; }

private:
    FrameRate m_frame_rate;
    uint32_t m_frame_rate_int;
    double m_ns_per_frame;
    uint32_t m_timecode_divisor;
    uint32_t m_timecode_fps;
    uint64_t m_start_time_ns;
    uint32_t m_current_frame;
    std::vector<TimecodePayloadType> m_payload_types;

    static constexpr uint16_t VITC_LINE_NUMBER = 9;
    static constexpr uint16_t LTC_LINE_NUMBER = 10;
    static constexpr uint16_t HORIZONTAL_OFFSET = 1296;
    static constexpr uint8_t USER_DATA_WORDS = 16;

    /**
     * @brief: Encodes S12M-2 timecode.
     *
     * @param [out] buffer: Buffer to store the encoded timecode.
     */
    void encode_s12m2_timecode(uint8_t* buffer);
    /**
     * @brief: Applies payload marker for the given payload type.
     *
     * @param [out] buffer: Buffer to apply payload marker to.
     * @param [in] payload_type: Payload type.
     */
    void apply_payload_marker(uint8_t* buffer, TimecodePayloadType payload_type);
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_MEDIA_TIMECODE_ENCODER_H_ */
