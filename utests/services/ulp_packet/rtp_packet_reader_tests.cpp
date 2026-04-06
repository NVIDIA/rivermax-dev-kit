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

#include "rdk/services/ulp_packet/readers/rtp_packet_reader.h"
#include "rdk/services/ulp_packet/rtp_header.h"
#include "rdk/services/ulp_packet/rtp_packet_context.h"
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
 * @brief: Stream the RTPPacketContext for GTest output in a readable format.
 *
 * @param [in] os: The output stream.
 * @param [in] context: The RTPPacketContext to stream.
 *
 * @return: The output stream.
 */
inline std::ostream& operator<<(std::ostream& os, const RTPPacketContext& context)
{
    return os << "(" << static_cast<int>(context.version) << ", " << static_cast<int>(context.payload_type)
              << ", " << (context.marker ? "true" : "false") << ", " << (context.padding ? "true" : "false")
              << ", " << (context.extension ? "true" : "false") << ", " << static_cast<int>(context.cc)
              << ", " << context.sequence << ", " << context.timestamp << ", " << context.ssrc << ")";
}

} // namespace services
} // namespace rdk

/**
 * @brief: Build expected RTPPacketContext from header fields (RTPPacketContext has a base class, so no aggregate init).
 *
 * @param [in] version: The RTP version.
 * @param [in] payload_type: The RTP payload type.
 * @param [in] marker: The RTP marker flag.
 * @param [in] padding: The RTP padding flag.
 * @param [in] extension: The RTP extension flag.
 * @param [in] cc: The RTP CSRC count.
 * @param [in] sequence: The RTP sequence number.
 * @param [in] timestamp: The RTP timestamp.
 * @param [in] ssrc: The RTP synchronization source identifier.
 *
 * @return: The expected RTPPacketContext.
 */
static RTPPacketContext make_expected_rtp_context(uint8_t version, uint8_t payload_type, bool marker,
                                                  bool padding, bool extension, uint8_t cc,
                                                  uint16_t sequence, uint32_t timestamp, uint32_t ssrc)
{
    RTPPacketContext context;
    context.version = version;
    context.payload_type = payload_type;
    context.marker = marker;
    context.padding = padding;
    context.extension = extension;
    context.cc = cc;
    context.sequence = sequence;
    context.timestamp = Rational(timestamp);
    context.ssrc = ssrc;
    return context;
}

/**
 * @brief: Test parameter for RTP packet reader tests.
 *
 * This struct is used to store the test parameters for the RTP packet reader tests.
 */
struct RTPParseHeaderTestParam
{
    std::string name;                             /**< Test name */
    std::array<uint8_t, RTP_HEADER_SIZE> buffer;  /**< Buffer with RTP header */
    RTPPacketContext expected;                    /**< Expected RTP header fields */
    /**
     * @brief: Stream the RTPParseHeaderTestParam for GTest output in a readable format.
     *
     * @param [in] os: The output stream.
     * @param [in] p: The RTPParseHeaderTestParam to stream.
     *
     * @return: The output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const RTPParseHeaderTestParam& p)
    {
        return os << p.expected;
    }
};

static std::string ParseHeaderTestName(const testing::TestParamInfo<RTPParseHeaderTestParam>& info)
{
    return info.param.name;
}

/**
 * @brief: Test helper for RTP packet reader tests.
 */
struct RTPPacketReaderTestHelper
{
protected:
    std::unique_ptr<uint8_t[]> m_buffer;
    byte_t* m_header_ptr = nullptr;
    /**
     * @brief: Setup the test helper.
     */
    void setup()
    {
        m_buffer = std::make_unique<uint8_t[]>(RTP_HEADER_SIZE);
        m_header_ptr = m_buffer.get();
    }
    /**
     * @brief: Parse the header from a buffer and populate the context.
     *
     * @param [in] buffer: The buffer with RTP header and expected header fields.
     * @param [out] context: The context to populate with the parsed RTP header fields.
     */
    void parse_header_from_buffer(const std::array<uint8_t, RTP_HEADER_SIZE>& buffer, RTPPacketContext& context)
    {
        memcpy(m_header_ptr, buffer.data(), RTP_HEADER_SIZE);
        RTPPacketReader reader(m_header_ptr, nullptr);
        ASSERT_EQ(reader.parse_header(context), ReturnStatus::success);
    }
    /**
     * @brief: Expect the context to match the expected header fields.
     *
     * @param [in] expected: The expected header fields.
     * @param [in] context: The context containing the parsed RTP header fields.
     */
    void expect_header_fields_match(const RTPPacketContext& expected,
                                    const RTPPacketContext& parsed_context)
    {
        EXPECT_EQ(parsed_context.version, expected.version);
        EXPECT_EQ(parsed_context.payload_type, expected.payload_type);
        EXPECT_EQ(parsed_context.marker, expected.marker);
        EXPECT_EQ(parsed_context.padding, expected.padding);
        EXPECT_EQ(parsed_context.extension, expected.extension);
        EXPECT_EQ(parsed_context.cc, expected.cc);
        EXPECT_EQ(parsed_context.sequence, expected.sequence);
        EXPECT_EQ(parsed_context.timestamp, expected.timestamp);
        EXPECT_EQ(parsed_context.ssrc, expected.ssrc);
    }
};

/**
 * @brief: Test suite for base RTP packet reader (RFC 3550 fixed header).
 */
class RTPPacketReaderTest : public ::testing::Test, protected RTPPacketReaderTestHelper
{
protected:
    void SetUp() override { setup(); }
};

/**
 * @brief: Test suite for parsing the RTP header fields with parameterised test cases.
 */
class RTPPacketReaderParseHeaderTest : public ::testing::TestWithParam<RTPParseHeaderTestParam>,
                                       protected RTPPacketReaderTestHelper
{
protected:
    void SetUp() override { setup(); }
};

/* Test parsing the RTP header fields and expect the context to match the expected header fields. */
TEST_P(RTPPacketReaderParseHeaderTest, HeaderFieldsMatchExpected)
{
    const auto& param = GetParam();
    RTPPacketContext context;
    parse_header_from_buffer(param.buffer, context);
    expect_header_fields_match(param.expected, context);
}

/* Instantiate the test suite for parsing the RTP header fields with parameterised test cases. */
INSTANTIATE_TEST_SUITE_P(
    RTPPacketReaderTestParams,
    RTPPacketReaderParseHeaderTest,
    testing::Values(
        RTPParseHeaderTestParam{"Basic",
            {0x80, 0x60, 0x01, 0x02, 0x00, 0x00, 0x03, 0xE8, 0x11, 0x22, 0x33, 0x44},
            make_expected_rtp_context(2, 96, false, false, false, 0, 258, 1000, 0x11223344u)},
        RTPParseHeaderTestParam{"Marker",
            {0x80, 0xE0, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01},
            make_expected_rtp_context(2, 96, true, false, false, 0, 1, 0, 1)},
        RTPParseHeaderTestParam{"Extension",
            {0x90, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
            make_expected_rtp_context(2, 96, false, false, true, 0, 0, 0, 0)},
        RTPParseHeaderTestParam{"Padding",
            {0xA0, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
            make_expected_rtp_context(2, 96, false, true, false, 0, 0, 0, 0)}
    ),
    ParseHeaderTestName
);

/* Test getting the header size and expect it to be the size of the RTP header. */
TEST_F(RTPPacketReaderTest, GetHeaderSize_Returns12)
{
    RTPPacketReader reader(m_header_ptr, nullptr);
    EXPECT_EQ(reader.get_header_size(), RTP_HEADER_SIZE);
}

/* Test parsing the RTP payload and expect it to return success. */
TEST_F(RTPPacketReaderTest, ParsePayload_ReturnsSuccess)
{
    RTPPacketReader reader(m_header_ptr, nullptr);
    RTPPacketContext context;
    EXPECT_EQ(reader.parse_payload(context), ReturnStatus::success);
}
