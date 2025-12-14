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

#include <cstring>

#include "rdk/services/media/timecode_encoder.h"
#include "rdk/services/media/media_defs.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

TimecodeEncoder::TimecodeEncoder(
    const FrameRate& frame_rate,
    uint64_t start_time_ns,
    const std::vector<TimecodePayloadType>& payload_types) :
    m_frame_rate(frame_rate),
    m_frame_rate_int(rational_cast<uint32_t>(frame_rate)),
    m_ns_per_frame(rational_cast<double>(NS_IN_SEC / frame_rate)),
    m_start_time_ns(start_time_ns),
    m_current_frame(0),
    m_payload_types(payload_types)
{
    // Pre-calculate timecode fps values (SMPTE max 30 fps, higher fps repeat frames)
    constexpr uint32_t MAX_TIMECODE_FPS = 30;
    m_timecode_divisor = (m_frame_rate_int > MAX_TIMECODE_FPS)
        ? (m_frame_rate_int + MAX_TIMECODE_FPS - 1) / MAX_TIMECODE_FPS
        : 1;
    m_timecode_fps = m_frame_rate_int / m_timecode_divisor;
}

size_t TimecodeEncoder::get_packet_count() const
{
    return m_payload_types.size();
}

size_t TimecodeEncoder::write_data(size_t packet_index, uint8_t* buffer)
{
    // Calculate timecode based on current frame and start time
    uint64_t total_time_ns = m_start_time_ns + static_cast<uint64_t>(m_current_frame * m_ns_per_frame);
    uint32_t total_seconds = total_time_ns / NS_IN_SEC;
    uint32_t hours = (total_seconds / 3600) % 24;
    uint32_t minutes = (total_seconds % 3600) / 60;
    uint32_t seconds = total_seconds % 60;

    // Frame numbers repeated for higher fps
    uint32_t frames = (m_current_frame / m_timecode_divisor) % m_timecode_fps;

    // S12M-2 format is 16 bytes, BCD encoded
    // As defined in SMPTE ST 12, section 9, table 8.
    uint8_t timecode_buffer[USER_DATA_WORDS];

    // Byte 0-3: Frames
    timecode_buffer[0] = (frames % 10) << 4;
    timecode_buffer[1] = 0x00;
    timecode_buffer[2] = ((frames / 10) + 8) << 4;
    timecode_buffer[3] = 0x00;

    // Byte 4-7: Seconds
    timecode_buffer[4] = (seconds % 10) << 4;
    timecode_buffer[5] = 0x00;
    timecode_buffer[6] = (seconds / 10) << 4;
    timecode_buffer[7] = 0x00;

    // Byte 8-11: Minutes
    timecode_buffer[8] = (minutes % 10) << 4;
    timecode_buffer[9] = 0x00;
    timecode_buffer[10] = (minutes / 10) << 4;
    timecode_buffer[11] = 0x00;

    // Byte 12-15: Hours
    timecode_buffer[12] = (hours % 10) << 4;
    timecode_buffer[13] = 0x00;
    timecode_buffer[14] = (hours / 10) << 4;
    timecode_buffer[15] = 0x00;

    // Apply payload type marker for this packet
    apply_payload_marker(timecode_buffer, m_payload_types[packet_index]);

    std::memcpy(buffer, timecode_buffer, USER_DATA_WORDS);
    return USER_DATA_WORDS;
}

void TimecodeEncoder::apply_payload_marker(uint8_t* buffer, TimecodePayloadType payload_type)
{
    // Apply payload marker for the given payload type
    // As defined in SMPTE ST 12, section 5, figure 1.
    switch (payload_type) {
        case TimecodePayloadType::LTC:
            // LTC: bits in bytes 0-1 = 0000
            buffer[0] &= 0xF0;
            buffer[1] = 0x00;
            break;

        case TimecodePayloadType::VITC1:
            // VITC1: bits in byte 0 = 1000, byte 1 = 0000
            buffer[0] = (buffer[0] & 0xF0) | 0x08;
            buffer[1] = 0x00;
            break;

        case TimecodePayloadType::VITC2:
            // VITC2: bits in byte 0 = 0000, byte 1 = 1000
            buffer[0] &= 0xF0;
            buffer[1] = 0x08;
            break;
    }
}

void TimecodeEncoder::fill_descriptor_header(size_t packet_index, AncillaryDataHeader& header) const
{
    NOT_IN_USE(packet_index);

    header.did = ANCILLARY_TIMECODE_IDENTIFIER.did;
    header.sdid = ANCILLARY_TIMECODE_IDENTIFIER.sdid;
    header.line_number = (m_payload_types[packet_index] == TimecodePayloadType::LTC) ? LTC_LINE_NUMBER
                                                                                     : VITC_LINE_NUMBER;
    header.horizontal_offset = HORIZONTAL_OFFSET;
}

void TimecodeEncoder::reset()
{
    m_current_frame = 0;
}

} // namespace services
} // namespace dev_kit
} // namespace rivermax
