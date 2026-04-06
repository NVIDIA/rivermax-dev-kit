/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>

#include "rdk/services/ulp_packet/readers/rtp_smpte_2110_20_packet_reader.h"
#include "rdk/services/ulp_packet/rtp_header.h"
#include "rdk/services/ulp_packet/rtp_smpte_2110_20_extension.h"
#include "rdk/services/ulp_packet/rtp_smpte_2110_20_packet_context.h"
#include "rdk/services/ulp_packet/writers/rtp_smpte_2110_20_packet_writer.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/media/media_defs.h"
#include "rdk/services/utils/rational.h"

using namespace rdk::services;
using namespace testing;

/**
 * @brief: Build expected RTP_SMPTE_2110_20_PacketContext from header fields (context has base class, so no aggregate init).
 *
 * @param [in] sequence: The RTP sequence number.
 * @param [in] timestamp: The RTP timestamp.
 * @param [in] ssrc: The RTP synchronization source identifier.
 * @param [in] extended_sequence_number: The extended sequence number.
 * @param [in] srd_length: The SRD length.
 * @param [in] line_number: The line number.
 * @param [in] srd_offset: The SRD offset.
 * @param [in] rtp_interlace_field_indicator: The RTP interlace field indicator.
 *
 * @return: The expected RTP_SMPTE_2110_20_PacketContext.
 */
static RTP_SMPTE_2110_20_PacketContext make_expected_rtp_smpte_2110_20_context(
    uint16_t sequence, uint32_t timestamp, uint32_t ssrc, uint32_t extended_sequence_number,
    uint16_t srd_length, uint16_t line_number, uint16_t srd_offset,
    uint8_t rtp_interlace_field_indicator)
{
    RTP_SMPTE_2110_20_PacketContext context;
    context.sequence = sequence;
    context.timestamp = Rational(timestamp);
    context.ssrc = ssrc;
    context.extended_sequence_number = extended_sequence_number;
    context.srd_length = srd_length;
    context.line_number = line_number;
    context.srd_offset = srd_offset;
    context.rtp_interlace_field_indicator = rtp_interlace_field_indicator;
    return context;
}

/**
 * @brief: Test helper for RTP SMPTE 2110-20 packet reader/writer tests.
 */
struct RTP_SMPTE_2110_20_PacketReaderWriterTestHelper
{
protected:
    /**
     * @brief: Verify the parsed RTP_SMPTE_2110_20_PacketContext fields match the expected values.
     *
     * @param [in] expected: The expected header fields.
     * @param [in] parsed_context: The context containing the parsed header fields.
     */
    static void verify_smpte_2110_20_fields(const RTP_SMPTE_2110_20_PacketContext& expected,
                                            const RTP_SMPTE_2110_20_PacketContext& parsed_context)
    {
        EXPECT_EQ(parsed_context.sequence, expected.sequence);
        EXPECT_EQ(parsed_context.timestamp, expected.timestamp);
        EXPECT_EQ(parsed_context.ssrc, expected.ssrc);
        EXPECT_EQ(parsed_context.extended_sequence_number, expected.extended_sequence_number);
        EXPECT_EQ(parsed_context.srd_length, expected.srd_length);
        EXPECT_EQ(parsed_context.line_number, expected.line_number);
        EXPECT_EQ(parsed_context.srd_offset, expected.srd_offset);
        EXPECT_EQ(parsed_context.rtp_interlace_field_indicator, expected.rtp_interlace_field_indicator);
    }
};

/**
 * @brief: Test suite for RTP SMPTE 2110-20 packet reader/writer round-trip tests.
 */
class RTP_SMPTE_2110_20_PacketReaderWriterTest : public ::testing::Test,
                                                 protected RTP_SMPTE_2110_20_PacketReaderWriterTestHelper
{
protected:
    std::unique_ptr<uint8_t[]> m_buffer;
    void SetUp() override { m_buffer = std::make_unique<uint8_t[]>(RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE); }
};

/* Test that writing and reading the RTP SMPTE 2110-20 header produces fields that match the original written values. */
TEST_F(RTP_SMPTE_2110_20_PacketReaderWriterTest, RoundTrip_FieldsMatch)
{
    const RTP_SMPTE_2110_20_PacketContext write_context =
        make_expected_rtp_smpte_2110_20_context(100, 7200, 0x0A0B0C0Du, (1u << 16) | 100u, 1200, 42, 0, 0);

    RTP_SMPTE_2110_20_PacketWriter writer(m_buffer.get(), nullptr);
    size_t size = 0;
    ASSERT_EQ(writer.fill_header(write_context, size, nullptr), ReturnStatus::success);

    RTP_SMPTE_2110_20_PacketContext read_context;
    RTP_SMPTE_2110_20_PacketReader reader(m_buffer.get(), nullptr);
    ASSERT_EQ(reader.parse_header(read_context), ReturnStatus::success);

    verify_smpte_2110_20_fields(write_context, read_context);
}
