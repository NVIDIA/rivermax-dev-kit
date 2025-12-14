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

#ifndef RDK_SERVICES_MEDIA_CLOSED_CAPTION_608_ENCODER_H_
#define RDK_SERVICES_MEDIA_CLOSED_CAPTION_608_ENCODER_H_

#include <cstdint>
#include <queue>
#include <string>
#include <utility>

#include "rdk/services/media/ancillary_data_encoder_interface.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: CEA-608 closed caption encoder for SMPTE 2110-40.
 *
 * This class manages closed caption encoding for Pop-On mode CEA-608 captions.
 * It converts text strings into a sequence of CEA-608 commands wrapped in
 * CEA-708 CDP (Caption Distribution Packet) structure.
 *
 * Usage:
 *   1. Each frame, call update(text) with the desired caption text
 *   2. Call write_data(packet_index, buffer) to write the CDP packet for this frame
 *      Even if no text should be displayed, write_data() must be called to produce a packet.
 *
 * The encoder automatically manages:
 *   - Text change detection (builds new caption when text changes)
 *   - Caption build sequence (RCL, ENM, PAC, TEXT, EOC)
 *   - Caption display (sends padding while text unchanged)
 *   - Screen clearing (EDM when text becomes empty)
 *   - CDP packet structure and parity bits
 *
 * It is based on the CTA-708 and SMPTE ST 334 specifications.
 */
class ClosedCaption608Encoder : public IAncillaryDataEncoder
{
public:
    /**
     * @brief: Constructs a CEA-608 encoder.
     *
     * @param [in] frame_rate: Frame rate.
     */
    explicit ClosedCaption608Encoder(const FrameRate& frame_rate);
    size_t write_data(size_t packet_index, uint8_t* buffer) override;
    void fill_descriptor_header(size_t packet_index, AncillaryDataHeader& header) const override;
    void reset() override;
    /**
     * @brief: Updates the desired caption text for this frame.
     *
     * Call this once per frame with the desired text.
     * The encoder will detect text changes and trigger the appropriate
     * caption sequence.
     *
     * @param [in] text: The text to display.
     */
    void update(const std::string& text);

private:
    /** SMPTE 2110-40 ancillary data descriptor */
    /* Line number for ancillary data descriptor */
    static constexpr uint8_t LINE_NUMBER = 10;
    /* Horizontal offset for ancillary data descriptor */
    static constexpr uint8_t HORIZONTAL_OFFSET = 0;

    /** CEA-608 channel command codes */
    /* Channel 1 command code */
    static constexpr uint8_t CMD_CH1 = 0x14;
    /* Channel 2 command code */
    static constexpr uint8_t CMD_CH2 = 0x15;

    /** CEA-608 control codes */
    /* Resume Caption Loading control code */
    static constexpr uint8_t RCL = 0x20;
    /* Erase Non-displayed Memory control code */
    static constexpr uint8_t ENM = 0x2E;
    /* End of Caption control code */
    static constexpr uint8_t EOC = 0x2F;
    /* Erase Displayed Memory control code */
    static constexpr uint8_t EDM = 0x2C;
    /* Preamble Address Code for row 15, indent 0 */
    static constexpr uint8_t PAC_ROW15_INDENT0 = 0x60;
    /* Padding byte */
    static constexpr uint8_t PADDING = 0x80;

    /** CEA-608 encoding limits */
    /* Maximum characters per frame */
    static constexpr size_t MAX_CHARS_PER_FRAME = 2;

    /** CEA-708 CDP packet constants */
    /* CDP identifier */
    static constexpr uint16_t CDP_IDENTIFIER = 0x9669;
    /* CDP flags: data present and caption service active */
    static constexpr uint8_t CDP_FLAGS = 0x43;
    /* CDP data type: CEA-608 data */
    static constexpr uint8_t CDP_CCDATA_ID_608 = 0x72;
    /* CDP CC count: 1 triplet (1 CEA-608 pair) */
    static constexpr uint8_t CDP_CC_COUNT = 0xE1;
    /* CDP CC valid flag */
    static constexpr uint8_t CDP_CC_VALID_FLAG = 0xFC;
    /* CDP footer ID */
    static constexpr uint8_t CDP_FOOTER_ID = 0x74;

    /** CEA-608 parity bit constants */
    /* Parity bit mask */
    static constexpr uint8_t PARITY_BIT_MASK = 0x7F;
    /* Parity bit */
    static constexpr uint8_t PARITY_BIT = 0x80;

    /**
     * @brief: State machine for caption display.
     */
    enum class State
    {
        /* No caption, no action needed */
        Idle,
        /* Detected new text, need to start building sequence */
        Start,
        /* Actively building caption (RCL, ENM, PAC, TEXT, EOC) */
        Building,
        /* Caption displayed, sending padding */
        Displaying,
        /* Detected text cleared, need to clear screen */
        StartingClear,
        /* Screen cleared, sending padding */
        Cleared,
    };
    State m_state;
    std::string m_active_text;
    std::string m_pending_text;
    std::queue<std::pair<uint8_t, uint8_t>> m_command_queue;
    uint8_t m_framerate_code;
    uint16_t m_sequence_counter;

    /**
     * @brief: Sets the state.
     *
     * @param [in] new_state: New state.
     */
    void set_state(State new_state);
    /**
     * @brief: Returns the frame data.
     *
     * @return: The frame data (Command code and/or data byte).
     */
    std::pair<uint8_t, uint8_t> get_frame_data();
    /**
     * @brief: Builds the caption sequence.
     */
    void build_caption_sequence(const std::string& text);
    /**
     * @brief: Builds the clear sequence.
     */
    void build_clear_sequence();
    /**
     * @brief: Encodes CEA-608 in CDP.
     *
     * @param [out] buffer: Buffer to store the encoded data.
     * @param [in] byte1: First byte of the CEA-608 data.
     * @param [in] byte2: Second byte of the CEA-608 data.
     *
     * @return: The number of bytes written to the buffer.
     */
    size_t encode_cea608_in_cdp(uint8_t* buffer, uint8_t byte1, uint8_t byte2);
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_CLOSED_CAPTION_608_ENCODER_H_ */
