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

#include <vector>

#include <gtest/gtest.h>

#include "rdk/services/ulp_packet/readers/rtp_smpte_2110_40_packet_reader.h"
#include "rdk/services/ulp_packet/writers/rtp_smpte_2110_40_packet_writer.h"
#include "rdk/services/ulp_packet/smpte_2110_40_ancillary_utils.h"
#include "rdk/services/media/media_defs.h"
#include "rtp_smpte_2110_40_test_utils.h"

using namespace rdk::services;
using namespace testing;

/* get_header_size() equals RTP 12 + extension 8 = 20 bytes. */
TEST(GetHeaderSizeTest, HeaderSize_EqualsAncillaryHeaderSizeConstant)
{
    const uint8_t dummy[20] = {};
    RTP_SMPTE_2110_40_PacketReader reader(dummy);
    EXPECT_EQ(reader.get_header_size(), RTP_ST_2110_40_ANCILLARY_HEADER_SIZE);
}

/**
 * @brief: Test fixture for AncillaryDataPacketReader::read_ancillary_data (single sub-packet header + 10-bit words).
 */
class ReadAncillaryDataTest : public ::testing::Test
{
protected:
    std::vector<uint8_t> m_wire;
    std::vector<uint8_t> m_out_data;
    AncillaryDataDescriptor m_out_descriptor;

    void SetUp() override
    {
        m_wire.clear();
        m_out_data.clear();
        m_out_descriptor = AncillaryDataDescriptor{};
    }

    size_t build_and_read(const AncillaryDataDescriptor& source_descriptor,
        std::vector<uint8_t> user_data_words_bytes,
        size_t current_offset = 0)
    {
        const size_t packet_size =
            AncillaryDataUtils::calculate_packet_size(
                source_descriptor.ancillary_data_header.user_data_words_count);
        m_wire.assign(packet_size, 0);

        AncillaryDataPacketWriter::write_ancillary_data(
            m_wire.data(),
            user_data_words_bytes.data(),
            source_descriptor);

        const size_t user_data_words_count = source_descriptor.ancillary_data_header.user_data_words_count;
        m_out_data.assign(std::max(static_cast<size_t>(1), user_data_words_count), 0);
        m_out_descriptor = AncillaryDataDescriptor{};

        return AncillaryDataPacketReader::read_ancillary_data(
            m_wire.data(), m_wire.size(), current_offset, m_out_descriptor, m_out_data.data());
    }
};

/* c_flag and s_flag round-trip independently for both true and false. */
TEST_F(ReadAncillaryDataTest, BooleanFlags_RoundTrip)
{
    build_and_read(make_descriptor(0x61, 0x01, 1, 0, 0, true, false), {0x01});
    EXPECT_TRUE(m_out_descriptor.ancillary_data_header.c_flag);
    EXPECT_FALSE(m_out_descriptor.ancillary_data_header.s_flag);

    build_and_read(make_descriptor(0x61, 0x01, 1, 0, 0, false, true), {0x01});
    EXPECT_FALSE(m_out_descriptor.ancillary_data_header.c_flag);
    EXPECT_TRUE(m_out_descriptor.ancillary_data_header.s_flag);
}

/* line_number 11-bit max 0x7FF round-trips. */
TEST_F(ReadAncillaryDataTest, LineNumber_MaxValue)
{
    build_and_read(make_descriptor(0x61, 0x01, 1, 0x7FF, 0), {0x01});
    EXPECT_EQ(m_out_descriptor.ancillary_data_header.line_number, 0x7FFu);
}

/* horizontal_offset 12-bit max 0xFFF round-trips. */
TEST_F(ReadAncillaryDataTest, HorizontalOffset_MaxValue)
{
    build_and_read(make_descriptor(0x61, 0x01, 1, 0, 0xFFF), {0x01});
    EXPECT_EQ(m_out_descriptor.ancillary_data_header.horizontal_offset, 0xFFFu);
}

/* stream_number 7-bit max 0x7F round-trips. */
TEST_F(ReadAncillaryDataTest, StreamNumber_MaxValue)
{
    build_and_read(make_descriptor(0x61, 0x01, 1, 0, 0, false, true, 0x7F), {0x01});
    EXPECT_EQ(m_out_descriptor.ancillary_data_header.stream_number, 0x7Fu);
}

/* DID and SDID both round-trip without aliasing each other. */
TEST_F(ReadAncillaryDataTest, DID_SDID_RoundTrip)
{
    build_and_read(make_descriptor(0xA5, 0x3C, 1), {0x01});
    EXPECT_EQ(m_out_descriptor.ancillary_data_header.did, 0xA5u);
    EXPECT_EQ(m_out_descriptor.ancillary_data_header.sdid, 0x3Cu);
}

/* user_data_words_count round-trips. */
TEST_F(ReadAncillaryDataTest, UserDataWordsCount_RoundTrips)
{
    build_and_read(make_descriptor(0x61, 0x01, 7), {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07});
    EXPECT_EQ(m_out_descriptor.ancillary_data_header.user_data_words_count, 7u);
}

/* user_data_offset is set to current_output_offset as passed in. */
TEST_F(ReadAncillaryDataTest, UserDataOffset_PassthroughCurrentOffset)
{
    build_and_read(make_descriptor(0x61, 0x01, 1), {0x01}, 0);
    EXPECT_EQ(m_out_descriptor.user_data_offset, 0u);

    build_and_read(make_descriptor(0x61, 0x01, 1), {0x01}, 42);
    EXPECT_EQ(m_out_descriptor.user_data_offset, 42u);
}

/* Four user data words exercise all bit-offset positions in packed stream. */
TEST_F(ReadAncillaryDataTest, FourUDWs_ExercisesAllBitOffsets)
{
    const std::vector<uint8_t> user_data_words = {0x11, 0x22, 0x33, 0x44};
    build_and_read(make_descriptor(0x61, 0x01, 4), user_data_words);
    EXPECT_EQ(m_out_data[0], 0x11u);
    EXPECT_EQ(m_out_data[1], 0x22u);
    EXPECT_EQ(m_out_data[2], 0x33u);
    EXPECT_EQ(m_out_data[3], 0x44u);
}

/* Five user data words span 32-bit boundary (word index 7, bit offset 2). */
TEST_F(ReadAncillaryDataTest, FiveUDWs_WrapsAcrossWordBoundary)
{
    const std::vector<uint8_t> user_data_words = {0xA1, 0xB2, 0xC3, 0xD4, 0xE5};
    build_and_read(make_descriptor(0x61, 0x01, 5), user_data_words);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(m_out_data[i], user_data_words[i]) << "byte " << i;
    }
}

/* Zero user data words: returns correct packet size and all descriptor fields decoded. */
TEST_F(ReadAncillaryDataTest, ZeroUDW_PacketSizeAndFields)
{
    const size_t bytes = build_and_read(make_descriptor(0xDE, 0xAD, 0, 100, 200, true, false, 3), {0});
    EXPECT_EQ(bytes, AncillaryDataUtils::calculate_packet_size(0));
    EXPECT_GT(bytes, 0u);

    const auto& hdr = m_out_descriptor.ancillary_data_header;
    EXPECT_EQ(hdr.did, 0xDEu);
    EXPECT_EQ(hdr.sdid, 0xADu);
    EXPECT_EQ(hdr.user_data_words_count, 0u);
    EXPECT_EQ(hdr.line_number, 100u);
    EXPECT_EQ(hdr.horizontal_offset, 200u);
    EXPECT_TRUE(hdr.c_flag);
    EXPECT_EQ(hdr.stream_number, 3u);
}

/* Corrupted packed word (parity/checksum) returns 0. */
TEST_F(ReadAncillaryDataTest, CorruptedPackedWord_ReturnsZero)
{
    const size_t pkt_size = AncillaryDataUtils::calculate_packet_size(2);
    m_wire.assign(pkt_size, 0);
    std::vector<uint8_t> user_data_words = {0x42, 0x43};
    AncillaryDataPacketWriter::write_ancillary_data(
        m_wire.data(), user_data_words.data(), make_descriptor(0x61, 0x01, 2));

    m_wire[RTP_ST_2110_40_FIXED_DATA_HEADER_SIZE] ^= 0x01;

    m_out_data.assign(4, 0);
    m_out_descriptor = AncillaryDataDescriptor{};
    const size_t result = AncillaryDataPacketReader::read_ancillary_data(
        m_wire.data(), m_wire.size(), 0, m_out_descriptor, m_out_data.data());
    EXPECT_EQ(result, 0u);
}

/* Flipping DID inverse-parity bit (bit 9) breaks parity, returns 0. */
TEST_F(ReadAncillaryDataTest, CorruptedInverseParity_ReturnsZero)
{
    const size_t pkt_size = AncillaryDataUtils::calculate_packet_size(2);
    m_wire.assign(pkt_size, 0);
    std::vector<uint8_t> user_data_words = {0x42, 0x43};
    AncillaryDataPacketWriter::write_ancillary_data(
        m_wire.data(), user_data_words.data(), make_descriptor(0x61, 0x01, 2));

    m_wire[RTP_ST_2110_40_FIXED_DATA_HEADER_SIZE] ^= 0x80;

    m_out_data.assign(4, 0);
    m_out_descriptor = AncillaryDataDescriptor{};
    const size_t result = AncillaryDataPacketReader::read_ancillary_data(
        m_wire.data(), m_wire.size(), 0, m_out_descriptor, m_out_data.data());
    EXPECT_EQ(result, 0u);
}

/* remaining_bytes < minimum size returns 0. */
TEST_F(ReadAncillaryDataTest, TruncatedPacket_TooSmallForMinimum_ReturnsZero)
{
    m_wire.assign(2, 0);
    m_out_data.assign(1, 0);
    const size_t result = AncillaryDataPacketReader::read_ancillary_data(
        m_wire.data(), m_wire.size(), 0, m_out_descriptor, m_out_data.data());
    EXPECT_EQ(result, 0u);
}

/* DATA_COUNT 8 but remaining_bytes too small returns 0. */
TEST_F(ReadAncillaryDataTest, TruncatedPacket_DataCountExceedsRemaining_ReturnsZero)
{
    constexpr uint16_t user_data_words_count = 8;
    m_wire.assign(AncillaryDataUtils::calculate_packet_size(user_data_words_count), 0);
    std::vector<uint8_t> user_data_words(user_data_words_count, 0x42);
    AncillaryDataPacketWriter::write_ancillary_data(
        m_wire.data(), user_data_words.data(), make_descriptor(0x61, 0x01, user_data_words_count));

    const size_t truncated = AncillaryDataUtils::calculate_packet_size(0);
    m_out_data.assign(user_data_words_count, 0);
    m_out_descriptor = AncillaryDataDescriptor{};
    const size_t result = AncillaryDataPacketReader::read_ancillary_data(
        m_wire.data(), truncated, 0, m_out_descriptor, m_out_data.data());
    EXPECT_EQ(result, 0u);
}

/* Bytes consumed equals calculate_packet_size for sampled user data words counts. */
TEST_F(ReadAncillaryDataTest, BytesConsumed_MatchesCalculatePacketSize)
{
    for (uint16_t n : {uint16_t(0), uint16_t(1), uint16_t(4), uint16_t(8), uint16_t(16)}) {
        std::vector<uint8_t> user_data_words(n, 0x5A);
        const size_t consumed = build_and_read(make_descriptor(0x61, 0x01, n), user_data_words);
        EXPECT_EQ(consumed, AncillaryDataUtils::calculate_packet_size(n))
            << "user_data_words_count=" << n;
    }
}
