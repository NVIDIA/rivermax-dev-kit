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

#include <array>
#include <cstring>
#include <ostream>
#include <string>

#include <gtest/gtest.h>

#include "rdk/services/ulp_packet/readers/rtp_smpte_2110_20_packet_reader.h"
#include "rdk/services/ulp_packet/rtp_header.h"
#include "rdk/services/ulp_packet/rtp_smpte_2110_20_extension.h"
#include "rdk/services/ulp_packet/rtp_smpte_2110_20_packet_context.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/media/media_defs.h"
#include "rdk/services/utils/rational.h"

using namespace rdk::services;
using namespace testing;

namespace rdk
{
namespace services
{

/**
 * @brief: Stream the RTP_SMPTE_2110_20_PacketContext for GTest output in a readable format.
 *
 * @param [in] os: The output stream.
 * @param [in] context: The RTP_SMPTE_2110_20_PacketContext to stream.
 *
 * @return: The output stream.
 */
inline std::ostream& operator<<(std::ostream& os, const RTP_SMPTE_2110_20_PacketContext& context)
{
    return os << "(" << context.sequence << ", " << context.timestamp << ", " << context.ssrc
              << ", " << context.extended_sequence_number << ", " << context.srd_length
              << ", " << context.line_number << ", " << context.srd_offset << ", "
              << static_cast<int>(context.rtp_interlace_field_indicator) << ")";
}

} // namespace services
} // namespace rdk

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
 * @brief: Test parameter for RTP SMPTE 2110-20 packet reader tests.
 *
 * This struct is used to store the test parameters for the RTP SMPTE 2110-20 packet reader tests.
 */
struct RTP_SMPTE_2110_20_ParseHeaderTestParam
{
    std::string name;                                                   /**< Test name */
    std::array<uint8_t, RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE> buffer;  /**< Buffer with RTP + SRD header */
    RTP_SMPTE_2110_20_PacketContext expected;                           /**< Expected header fields */
    /**
     * @brief: Stream the RTP_SMPTE_2110_20_ParseHeaderTestParam for GTest output in a readable format.
     *
     * @param [in] os: The output stream.
     * @param [in] p: The RTP_SMPTE_2110_20_ParseHeaderTestParam to stream.
     *
     * @return: The output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const RTP_SMPTE_2110_20_ParseHeaderTestParam& p)
    {
        return os << p.expected;
    }
};

/**
 * @brief: Get the test name.
 *
 * @param [in] info: The test parameter information.
 *
 * @return: The test name.
 */
static std::string ParseHeaderTestName(const testing::TestParamInfo<RTP_SMPTE_2110_20_ParseHeaderTestParam>& info)
{
    return info.param.name;
}

/**
 * @brief: Test helper for RTP SMPTE 2110-20 packet reader tests.
 */
struct RTP_SMPTE_2110_20_PacketReaderTestHelper
{
protected:
    std::unique_ptr<uint8_t[]> m_header_buffer;
    byte_t* m_header_ptr = nullptr;
    /**
     * @brief: Setup the test helper.
     */
    void setup()
    {
        m_header_buffer = std::make_unique<uint8_t[]>(RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE);
        m_header_ptr = m_header_buffer.get();
    }
    /**
     * @brief: Copy the buffer and parse the header.
     *
     * @param [in] buffer: The buffer with RTP header and expected header fields.
     * @param [out] context: The context to populate with the parsed RTP header fields.
     */
    void parse_header_from_buffer(const std::array<uint8_t, RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE>& buffer,
                                  RTP_SMPTE_2110_20_PacketContext& context)
    {
        memcpy(m_header_ptr, buffer.data(), RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE);
        RTP_SMPTE_2110_20_PacketReader reader(m_header_ptr, nullptr);
        ASSERT_EQ(reader.parse_header(context), ReturnStatus::success);
    }
    /**
     * @brief: Expect the context to match the expected header fields.
     *
     * @param [in] expected: The expected header fields.
     * @param [in] parsed_context: The context containing the parsed RTP header fields.
     */
    void expect_context_matches(const RTP_SMPTE_2110_20_PacketContext& expected,
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
 * @brief: Test suite for RTP SMPTE 2110-20 packet reader
 *
 * This test suite checks that the SMPTE 2110-20 packet header fields are extracted correctly from the buffer.
 */
class RTP_SMPTE_2110_20_PacketReaderTest : public ::testing::Test,
                                           protected RTP_SMPTE_2110_20_PacketReaderTestHelper
{
protected:
    void SetUp() override { setup(); }
};

/**
 * @brief: Test suite for parsing the RTP SMPTE 2110-20 header with parameterised test cases.
 */
class RTP_SMPTE_2110_20_PacketReaderParseHeaderTest
    : public ::testing::TestWithParam<RTP_SMPTE_2110_20_ParseHeaderTestParam>,
      protected RTP_SMPTE_2110_20_PacketReaderTestHelper
{
protected:
    void SetUp() override { setup(); }
};

/* Test parsing the SMPTE 2110-20 RTP header fields and expect the context to match the expected header fields. */
TEST_P(RTP_SMPTE_2110_20_PacketReaderParseHeaderTest, ParseHeader_ContextMatchesExpected)
{
    const auto& param = GetParam();
    RTP_SMPTE_2110_20_PacketContext context;
    parse_header_from_buffer(param.buffer, context);
    expect_context_matches(param.expected, context);
}

/* Instantiate the test suite for parsing the RTP SMPTE 2110-20 header with parameterised test cases. */
INSTANTIATE_TEST_SUITE_P(
    RTP_SMPTE_2110_20_PacketReaderTestParams,
    RTP_SMPTE_2110_20_PacketReaderParseHeaderTest,
    testing::Values(
        RTP_SMPTE_2110_20_ParseHeaderTestParam{"Basic",
            { 0x80, 0x60, 0x01, 0x02, 0x00, 0x00, 0x03, 0xE8, 0x11, 0x22, 0x33, 0x44,
              0x00, 0x01, 0x04, 0xB0, 0x00, 0x2A, 0x00, 0x00 },
            make_expected_rtp_smpte_2110_20_context(258, 1000, 0x11223344u, (1u << 16) | 258u, 1200, 42, 0, 0)
        },
        RTP_SMPTE_2110_20_ParseHeaderTestParam{"Basic2",
            { 0x80, 0x60, 0x00, 0x64, 0x00, 0x00, 0x1C, 0x20, 0x0A, 0x0B, 0x0C, 0x0D,
              0x00, 0x01, 0x04, 0xB0, 0x00, 0x2A, 0x00, 0x00 },
            make_expected_rtp_smpte_2110_20_context(100, 7200, 0x0A0B0C0Du, (1u << 16) | 100u, 1200, 42, 0, 0)
        },
        RTP_SMPTE_2110_20_ParseHeaderTestParam{"Basic3",
            { 0x80, 0x60, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
              0x00, 0x02, 0x00, 0x00, 0x80, 0x00, 0x00, 0x64 },
            make_expected_rtp_smpte_2110_20_context(0xFFFF, 0, 0, (2u << 16) | 0xFFFFu, 0, 0, 100, 1)
        }
    ),
    ParseHeaderTestName
);

/* Test getting the header size and expect it to be the size of the RTP header and SRD header. */
TEST_F(RTP_SMPTE_2110_20_PacketReaderTest, GetHeaderSize_RTP_SMPTE_2110_20_HeaderSize)
{
    RTP_SMPTE_2110_20_PacketReader reader(m_header_ptr, nullptr);
    EXPECT_EQ(reader.get_header_size(), RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE);
}

/* Test parsing the RTP SMPTE 2110-20 RTP payload and expect it to return success. */
TEST_F(RTP_SMPTE_2110_20_PacketReaderTest, ParsePayload_ReturnsSuccess)
{
    RTP_SMPTE_2110_20_PacketReader reader(m_header_ptr, nullptr);
    RTP_SMPTE_2110_20_PacketContext context;
    EXPECT_EQ(reader.parse_payload(context), ReturnStatus::success);
}
