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

#include <iostream>
#include <numeric>
#include <unordered_map>

#include "rdk/services/media/closed_caption_608_encoder.h"
#include "rdk/services/media/media_defs.h"

namespace rdk
{
namespace services
{

/**
 * @brief: CDP frame rate codes.
 *
 * As defined in SMPTE ST 334-2, section 5.2, table 3.
 *
 * @note: This map is used to convert frame rates to the corresponding CDP frame rate code.
 */
static const std::unordered_map<FrameRate, uint8_t> CDP_FRAME_RATE_CODES = {
    { { 24000, 1001 }, 0x01 },  /**< 23.976 fps */
    { { 24 }, 0x02 },           /**< 24 fps */
    { { 25 }, 0x03 },           /**< 25 fps */
    { { 30000, 1001 }, 0x04 },  /**< 29.97 fps */
    { { 30 }, 0x05 },           /**< 30 fps */
    { { 50 }, 0x06 },           /**< 50 fps */
    { { 60000, 1001 }, 0x07 },  /**< 59.94 fps */
    { { 60 }, 0x08 }            /**< 60 fps */
};
static constexpr uint8_t CDP_FRAME_RATE_CODE_UNKNOWN = 0x0F;

constexpr uint8_t ClosedCaption608Encoder::CMD_CH1;
constexpr uint8_t ClosedCaption608Encoder::RCL;
constexpr uint8_t ClosedCaption608Encoder::ENM;
constexpr uint8_t ClosedCaption608Encoder::EOC;
constexpr uint8_t ClosedCaption608Encoder::EDM;
constexpr uint8_t ClosedCaption608Encoder::PAC_ROW15_INDENT0;
constexpr uint8_t ClosedCaption608Encoder::PADDING;

ClosedCaption608Encoder::ClosedCaption608Encoder(const FrameRate& frame_rate) :
    m_state(State::Idle),
    m_active_text(),
    m_pending_text(),
    m_sequence_counter(0)
{
    auto it = CDP_FRAME_RATE_CODES.find(frame_rate);
    if (it != CDP_FRAME_RATE_CODES.end()) {
        m_framerate_code = it->second;
    } else {
        m_framerate_code = CDP_FRAME_RATE_CODE_UNKNOWN;
        std::cerr << "Warning: Unsupported frame rate " << frame_rate
                  << " for closed caption encoding, using unknown code 0x0F" << std::endl;
    }
}

void ClosedCaption608Encoder::update(const std::string& text)
{
    m_pending_text = text;

    if (m_state == State::Idle || m_state == State::Cleared) {
        // When no caption is displayed, check if there is new text to display
        if (!text.empty() && text != m_active_text) {
            build_caption_sequence(text);
            set_state(State::Building);
        }
    } else if (m_state == State::Displaying) {
        // When a caption is displayed, check if it should be cleared or replaced
        if (text.empty()) {
            build_clear_sequence();
            set_state(State::StartingClear);
        } else if (text != m_active_text) {
            build_caption_sequence(text);
            set_state(State::Building);
        }
    }
}

std::pair<uint8_t, uint8_t> ClosedCaption608Encoder::get_frame_data()
{
    // Command queue can contain commands or data bytes to send
    if (!m_command_queue.empty()) {
        auto cmd = m_command_queue.front();
        m_command_queue.pop();

        // When the command queue is empty, transition to the next state
        if (m_command_queue.empty()) {
            if (m_state == State::Building) {
                set_state(State::Displaying);
            } else if (m_state == State::StartingClear) {
                set_state(State::Cleared);
            }
        }

        return cmd;
    }

    // When no command or data to send, return padding
    return { PADDING, PADDING };
}

void ClosedCaption608Encoder::reset()
{
    set_state(State::Idle);
    m_active_text.clear();
    m_pending_text.clear();

    while (!m_command_queue.empty()) {
        m_command_queue.pop();
    }
}

void ClosedCaption608Encoder::set_state(State new_state)
{
    m_state = new_state;
}

void ClosedCaption608Encoder::build_caption_sequence(const std::string& text)
{
    m_active_text = text;

    // Pop-On caption sequence: RCL, ENM, PAC, TEXT, EOC
    m_command_queue.push({ CMD_CH1, RCL });
    m_command_queue.push({ CMD_CH1, ENM });
    m_command_queue.push({ CMD_CH1, PAC_ROW15_INDENT0 });

    // For legacy reason, 608 captions send 2 characters per frame
    for (size_t i = 0; i < text.length(); i += 2) {
        uint8_t byte1 = static_cast<uint8_t>(text[i]);
        uint8_t byte2 = (i + 1 < text.length()) ? static_cast<uint8_t>(text[i + 1]) : PADDING;
        m_command_queue.push({ byte1, byte2 });
    }

    m_command_queue.push({ CMD_CH1, EOC });
}

void ClosedCaption608Encoder::build_clear_sequence()
{
    // Tell the receiver to clear the screen
    m_command_queue.push({ CMD_CH1, EDM });
    m_active_text.clear();
}

size_t ClosedCaption608Encoder::write_data(size_t packet_index, uint8_t* buffer)
{
    NOT_IN_USE(packet_index);

    // Get next queued command or data byte and encode it
    auto frame_data = get_frame_data();
    return encode_cea608_in_cdp(buffer, frame_data.first, frame_data.second);
}

void ClosedCaption608Encoder::fill_descriptor_header(size_t packet_index, AncillaryDataHeader& header) const
{
    NOT_IN_USE(packet_index);

    header.did = ANCILLARY_CLOSED_CAPTION_IDENTIFIER.did;
    header.sdid = ANCILLARY_CLOSED_CAPTION_IDENTIFIER.sdid;
    header.line_number = LINE_NUMBER;
    header.horizontal_offset = HORIZONTAL_OFFSET;
}

size_t ClosedCaption608Encoder::encode_cea608_in_cdp(uint8_t* buffer, uint8_t byte1, uint8_t byte2)
{
    // Encode CEA-608 data in CEA-708 CDP (Caption Distribution Packet)

    // Helper function to add parity bit to a byte
    auto add_parity = [](uint8_t byte_value) -> uint8_t {
        byte_value &= PARITY_BIT_MASK;
        uint8_t parity = 0;
        for (int bit_index = 0; bit_index < 7; ++bit_index) {
            if (byte_value & (1 << bit_index))
                parity++;
        }
        if ((parity & 1) == 0)
            byte_value |= PARITY_BIT;
        return byte_value;
    };

    size_t offset = 0;

    // CDP Header
    buffer[offset++] = (CDP_IDENTIFIER >> 8) & 0xFF;
    buffer[offset++] = CDP_IDENTIFIER & 0xFF;

    // Keep lenght offset
    size_t length_offset = offset;
    buffer[offset++] = 0x00;

    // CDP Header
    buffer[offset++] = (m_framerate_code << 4) | 0x0F;
    buffer[offset++] = CDP_FLAGS;
    buffer[offset++] = (m_sequence_counter >> 8) & 0xFF;
    buffer[offset++] = m_sequence_counter & 0xFF;

    // CEA-608 data
    buffer[offset++] = CDP_CCDATA_ID_608;
    buffer[offset++] = CDP_CC_COUNT;
    buffer[offset++] = CDP_CC_VALID_FLAG;
    buffer[offset++] = add_parity(byte1);
    buffer[offset++] = add_parity(byte2);

    // CDP Footer
    buffer[offset++] = CDP_FOOTER_ID;
    buffer[offset++] = (m_sequence_counter >> 8) & 0xFF;
    buffer[offset++] = m_sequence_counter & 0xFF;

    // Keep checksum offset
    size_t checksum_offset = offset;
    buffer[offset++] = 0x00;

    // Backfill length
    buffer[length_offset] = static_cast<uint8_t>(offset);

    // Calculate checksum and backfill
    uint8_t checksum = std::accumulate(buffer, buffer + offset, static_cast<uint8_t>(0));
    buffer[checksum_offset] = static_cast<uint8_t>(256 - checksum);

    m_sequence_counter++;
    return offset;
}

} // namespace services
} // namespace rdk
