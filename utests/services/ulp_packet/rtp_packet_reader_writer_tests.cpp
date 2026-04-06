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

#include <iostream>

#include <gtest/gtest.h>

#include "rdk/services/ulp_packet/readers/rtp_packet_reader.h"
#include "rdk/services/ulp_packet/rtp_header.h"
#include "rdk/services/ulp_packet/rtp_packet_context.h"
#include "rdk/services/ulp_packet/writers/rtp_packet_writer.h"
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
 * @brief: Test helper for RTP packet reader/writer tests.
 */
struct RTPPacketReaderWriterTestHelper
{
protected:
    std::unique_ptr<uint8_t[]> m_buffer;
    /**
     * @brief: Setup the test helper.
     */
    void setup()
    {
        m_buffer = std::make_unique<uint8_t[]>(RTP_HEADER_SIZE);
    }
    /**
     * @brief: Verify the parsed RTPPacketContext fields match the expected values.
     *
     * @param [in] expected: The expected header fields.
     * @param [in] parsed_context: The context containing the parsed RTP header fields.
     */
    void verify_rtp_header_fields(const RTPPacketContext& expected,
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
 * @brief: Test suite for base RTP packet reader/writer round-trip tests.
 */
class RTPPacketReaderWriterTest : public ::testing::TestWithParam<RTPPacketContext>,
                                  protected RTPPacketReaderWriterTestHelper
{
protected:
    void SetUp() override { setup(); }
};

/* Test writing the RTP header and reading it back and expect the source fields to match. */
TEST_P(RTPPacketReaderWriterTest, RoundTrip_FieldsMatch)
{
    const RTPPacketContext write_context = GetParam();

    RTPPacketWriter writer(m_buffer.get(), nullptr);
    size_t size = 0;
    ASSERT_EQ(writer.fill_header(write_context, size, nullptr), ReturnStatus::success);
    ASSERT_EQ(size, RTP_HEADER_SIZE);

    RTPPacketContext read_context;
    RTPPacketReader reader(m_buffer.get(), nullptr);
    ASSERT_EQ(reader.parse_header(read_context), ReturnStatus::success);

    verify_rtp_header_fields(write_context, read_context);
}

/* Instantiate the test suite for base RTP packet reader/writer round-trip with parameterised cases. */
INSTANTIATE_TEST_SUITE_P(
    BaseRTP,
    RTPPacketReaderWriterTest,
    testing::Values(
        make_expected_rtp_context(2, 96, true, false, false, 0, 1000, 3600, 0x01020304u),
        make_expected_rtp_context(2, 96, false, false, false, 0, 0, 0, 0),
        make_expected_rtp_context(2, 127, false, false, false, 0, 0x1234, 1000, 0x11223344u)
    ));
