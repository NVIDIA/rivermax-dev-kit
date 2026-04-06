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
#include "rdk/services/ulp_packet/rtp_smpte_2110_40_packet_context.h"
#include "rdk/services/media/media_defs.h"
#include "rdk/services/error_handling/return_status.h"
#include "rtp_smpte_2110_40_test_utils.h"

using namespace rdk::services;
using namespace testing;

/**
 * @brief: Test suite for full RTP packet round-trip (write then read; descriptor and user data words byte fidelity).
 */
class FullRoundTripTest : public TestWithParam<uint16_t> {};

/* Roundtrip: descriptor and user data words bytes match for parameterized user_data_words count. */
TEST_P(FullRoundTripTest, DataFidelity_DescriptorAndUDWBytes)
{
    const uint16_t user_data_words_count = GetParam();

    const AncillaryDataDescriptor source_descriptor =
        make_descriptor(0x61, 0x02, user_data_words_count, 15, 100, true, false, 3, 0);

    const size_t user_data_words_size = std::max(static_cast<size_t>(1), static_cast<size_t>(user_data_words_count));
    std::vector<uint8_t> source_user_data_words(user_data_words_size);
    for (uint16_t i = 0; i < user_data_words_count; ++i) {
        source_user_data_words[i] = static_cast<uint8_t>((i * 37 + 13) % 256);
    }

    std::vector<uint8_t> buffer;
    build_rtp_packet(buffer, {source_descriptor}, source_user_data_words);

    std::vector<AncillaryDataDescriptor> output_descriptors;
    std::vector<uint8_t> output_data(std::max(static_cast<size_t>(1),
        static_cast<size_t>(user_data_words_count)), 0);

    RTP_SMPTE_2110_40_PacketContext context;
    context.descriptors = &output_descriptors;
    context.payload_ptr = output_data.data();

    RTP_SMPTE_2110_40_PacketReader reader(buffer.data(), nullptr);
    ASSERT_EQ(reader.parse_header(context), ReturnStatus::success);
    ASSERT_EQ(reader.parse_payload(context), ReturnStatus::success);

    ASSERT_EQ(output_descriptors.size(), 1u);
    EXPECT_EQ(context.descriptor_count_in_packet, 1u);

    const auto& header = output_descriptors[0].ancillary_data_header;
    EXPECT_EQ(header.did, source_descriptor.ancillary_data_header.did);
    EXPECT_EQ(header.sdid, source_descriptor.ancillary_data_header.sdid);
    EXPECT_EQ(header.user_data_words_count, user_data_words_count);
    EXPECT_EQ(header.line_number, source_descriptor.ancillary_data_header.line_number);
    EXPECT_EQ(header.horizontal_offset, source_descriptor.ancillary_data_header.horizontal_offset);
    EXPECT_EQ(header.c_flag, source_descriptor.ancillary_data_header.c_flag);
    EXPECT_EQ(header.stream_number, source_descriptor.ancillary_data_header.stream_number);

    for (uint16_t i = 0; i < user_data_words_count; ++i) {
        EXPECT_EQ(output_data[i], source_user_data_words[i]) << "user_data_word[" << i << "]";
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllUDWCounts,
    FullRoundTripTest,
    Values(uint16_t(0),  uint16_t(1),  uint16_t(2),  uint16_t(3),
        uint16_t(4),  uint16_t(7),  uint16_t(8),  uint16_t(15),
        uint16_t(16), uint16_t(32), uint16_t(64), uint16_t(128),
        uint16_t(255)));

/**
 * @brief: Test fixture for full RTP packet parsing (parse_header and parse_payload; multiple sub-packets and edge cases).
 *
 * Packets are built with the writer; behavior under corruption, partial failure, and accumulation
 * patterns is verified on the reader side.
 */
class ParsePayloadTest : public ::testing::Test
{
protected:
    std::vector<uint8_t> m_packet_buffer;
    std::vector<uint8_t> m_user_data_words_bytes;
    std::vector<AncillaryDataDescriptor> m_out_descriptors;
    std::vector<uint8_t> m_out_data;

    void SetUp() override
    {
        m_user_data_words_bytes = {0xAB, 0xCD, 0xEF, 0x12};
        m_out_descriptors.clear();
        m_out_data.assign(4096, 0);
    }

    RTP_SMPTE_2110_40_PacketContext build_and_parse(
        uint16_t sequence, uint32_t timestamp, uint32_t ssrc, uint32_t extended_sequence,
        uint8_t field_indicator, uint32_t ancillary_count = 1)
    {
        std::vector<AncillaryDataDescriptor> descriptors;
        std::vector<uint8_t> user_data_words_bytes;
        for (uint32_t i = 0; i < ancillary_count; ++i) {
            descriptors.push_back(make_descriptor(0x61, 0x01,
                static_cast<uint16_t>(m_user_data_words_bytes.size())));
            user_data_words_bytes.insert(user_data_words_bytes.end(), m_user_data_words_bytes.begin(), m_user_data_words_bytes.end());
        }
        build_rtp_packet(m_packet_buffer, descriptors, user_data_words_bytes, sequence, timestamp, ssrc,
            extended_sequence, field_indicator);

        RTP_SMPTE_2110_40_PacketContext context;
        context.descriptors = &m_out_descriptors;
        context.payload_ptr = m_out_data.data();

        RTP_SMPTE_2110_40_PacketReader reader(m_packet_buffer.data(), nullptr);
        ReturnStatus header_status = reader.parse_header(context);
        EXPECT_EQ(header_status, ReturnStatus::success);
        ReturnStatus payload_status = reader.parse_payload(context);
        EXPECT_EQ(payload_status, ReturnStatus::success);
        return context;
    }

    void build(const std::vector<AncillaryDataDescriptor>& descriptors,
        std::vector<uint8_t> user_data_words_bytes = {})
    {
        build_rtp_packet(m_packet_buffer, descriptors, std::move(user_data_words_bytes));
    }

    ReturnStatus parse(RTP_SMPTE_2110_40_PacketContext& context)
    {
        if (context.descriptors == nullptr) {
            context.descriptors = &m_out_descriptors;
        }
        if (context.payload_ptr == nullptr) {
            context.payload_ptr = m_out_data.data();
        }
        RTP_SMPTE_2110_40_PacketReader reader(m_packet_buffer.data(), nullptr);
        ReturnStatus header_status = reader.parse_header(context);
        if (header_status != ReturnStatus::success) {
            return ReturnStatus::failure;
        }
        return reader.parse_payload(context);
    }
};

/* RTP version field decodes as 2. */
TEST_F(ParsePayloadTest, Version_Is2)
{
    const auto context = build_and_parse(1, 0, 0, 1,
        RTP_2110_40_FIELD_INDICATOR_PROGRESSIVE);
    EXPECT_EQ(context.version, 2u);
}

/* Sequence number round-trips. */
TEST_F(ParsePayloadTest, SequenceNumber_RoundTrips)
{
    const auto context = build_and_parse(0xBEEF, 0, 0, 0xBEEF,
        RTP_2110_40_FIELD_INDICATOR_PROGRESSIVE);
    EXPECT_EQ(context.sequence, 0xBEEFu);
}

/* Timestamp round-trips. */
TEST_F(ParsePayloadTest, Timestamp_RoundTrips)
{
    const auto context = build_and_parse(1, 0xDEADBEEFu, 0, 1,
        RTP_2110_40_FIELD_INDICATOR_PROGRESSIVE);
    EXPECT_EQ(context.timestamp, 0xDEADBEEFu);
}

/* SSRC round-trips. */
TEST_F(ParsePayloadTest, SSRC_RoundTrips)
{
    const auto context = build_and_parse(1, 0, 0xCAFEBABEu, 1,
        RTP_2110_40_FIELD_INDICATOR_PROGRESSIVE);
    EXPECT_EQ(context.ssrc, 0xCAFEBABEu);
}

/* extended_sequence_number: high 16 from extension, low 16 from RTP sequence. */
TEST_F(ParsePayloadTest, ExtendedSequenceNumber_FullValueAssembled)
{
    constexpr uint16_t SEQ  = 0x1234;
    constexpr uint32_t HIGH = 0x0005u;
    constexpr uint32_t EXT  = (HIGH << 16) | SEQ;
    const auto context = build_and_parse(SEQ, 0, 0, EXT,
        RTP_2110_40_FIELD_INDICATOR_PROGRESSIVE);
    EXPECT_EQ(context.extended_sequence_number, EXT);
}

/* ancillary_count round-trips. */
TEST_F(ParsePayloadTest, AncillaryCount_RoundTrips)
{
    const auto context = build_and_parse(1, 0, 0, 1,
        RTP_2110_40_FIELD_INDICATOR_PROGRESSIVE, 3);
    EXPECT_EQ(context.ancillary_count, 3u);
}

/* All three field indicator values (progressive, field1, field2) round-trip. */
TEST_F(ParsePayloadTest, FieldIndicator_AllValues_RoundTrip)
{
    for (const uint8_t fi : {RTP_2110_40_FIELD_INDICATOR_PROGRESSIVE,
                              RTP_2110_40_FIELD_INDICATOR_FIELD1,
                              RTP_2110_40_FIELD_INDICATOR_FIELD2}) {
        const auto context = build_and_parse(1, 0, 0, 1, fi);
        EXPECT_EQ(context.field_indicator, fi) << "field_indicator=" << static_cast<int>(fi);
    }
}

/* Zero ANC sub-packets: parse_payload returns success. */
TEST_F(ParsePayloadTest, ZeroAncCount_ReturnsSuccess)
{
    build({});
    RTP_SMPTE_2110_40_PacketContext context;
    ASSERT_EQ(parse(context), ReturnStatus::success);
    EXPECT_EQ(context.descriptor_count_in_packet, 0u);
    EXPECT_TRUE(m_out_descriptors.empty());
}

/* anc_count==0 leaves pre-existing descriptors intact. */
TEST_F(ParsePayloadTest, ZeroAncCount_PreExistingDescriptors_Unchanged)
{
    build({});
    AncillaryDataDescriptor sentinel{};
    sentinel.ancillary_data_header.did = 0xAB;
    m_out_descriptors = {sentinel, sentinel, sentinel};

    RTP_SMPTE_2110_40_PacketContext context;
    ASSERT_EQ(parse(context), ReturnStatus::success);

    ASSERT_EQ(m_out_descriptors.size(), 3u);
    for (const auto& descriptor : m_out_descriptors) {
        EXPECT_EQ(descriptor.ancillary_data_header.did, 0xABu);
    }
    EXPECT_EQ(context.descriptor_count_in_packet, 0u);
}

/* Single sub-packet: all header fields and user data words bytes decoded. */
TEST_F(ParsePayloadTest, SingleSubPacket_AllFieldsDecoded)
{
    const std::vector<uint8_t> user_data_words = {0x10, 0x20, 0x30};
    const auto descriptor = make_descriptor(0x61, 0x42, 3, 15, 200, true, false, 5);
    build({descriptor}, user_data_words);

    RTP_SMPTE_2110_40_PacketContext context;
    ASSERT_EQ(parse(context), ReturnStatus::success);

    ASSERT_EQ(m_out_descriptors.size(), 1u);
    EXPECT_EQ(context.descriptor_count_in_packet, 1u);

    const auto& hdr = m_out_descriptors[0].ancillary_data_header;
    EXPECT_EQ(hdr.did, 0x61u);
    EXPECT_EQ(hdr.sdid, 0x42u);
    EXPECT_EQ(hdr.user_data_words_count, 3u);
    EXPECT_EQ(hdr.line_number, 15u);
    EXPECT_EQ(hdr.horizontal_offset, 200u);
    EXPECT_TRUE(hdr.c_flag);
    EXPECT_FALSE(hdr.s_flag);
    EXPECT_EQ(hdr.stream_number, 5u);

    EXPECT_EQ(m_out_descriptors[0].user_data_offset, 0u);
    EXPECT_EQ(m_out_data[0], 0x10u);
    EXPECT_EQ(m_out_data[1], 0x20u);
    EXPECT_EQ(m_out_data[2], 0x30u);
}

/* Multiple sub-packets: all descriptors and user data words bytes decoded. */
TEST_F(ParsePayloadTest, MultipleSubPackets_AllDecoded)
{
    constexpr uint16_t user_data_words_per_packet = 4;
    const std::vector<AncillaryDataDescriptor> descriptors = {
        make_descriptor(0x41, 0x01, user_data_words_per_packet, 10, 0, false, false, 0,  0),
        make_descriptor(0x42, 0x02, user_data_words_per_packet, 20, 0, true,  false, 0,  4),
        make_descriptor(0x43, 0x03, user_data_words_per_packet, 30, 0, false, true,  0,  8),
    };
    const std::vector<uint8_t> user_data_words = {
        0x01, 0x02, 0x03, 0x04,
        0x11, 0x12, 0x13, 0x14,
        0x21, 0x22, 0x23, 0x24,
    };
    build(descriptors, user_data_words);

    RTP_SMPTE_2110_40_PacketContext context;
    ASSERT_EQ(parse(context), ReturnStatus::success);

    ASSERT_EQ(m_out_descriptors.size(), 3u);
    EXPECT_EQ(context.descriptor_count_in_packet, 3u);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(m_out_descriptors[i].ancillary_data_header.did,
            descriptors[i].ancillary_data_header.did)  << "i=" << i;
        EXPECT_EQ(m_out_descriptors[i].ancillary_data_header.sdid,
            descriptors[i].ancillary_data_header.sdid) << "i=" << i;
        EXPECT_EQ(m_out_descriptors[i].user_data_offset, i * user_data_words_per_packet) << "i=" << i;

        const size_t base = m_out_descriptors[i].user_data_offset;
        for (uint16_t j = 0; j < user_data_words_per_packet; ++j) {
            EXPECT_EQ(m_out_data[base + j], user_data_words[i * user_data_words_per_packet + j])
                << "i=" << i << " j=" << j;
        }
    }
}

/* Corrupted packed word on first sub-packet: parse_payload returns failure. */
TEST_F(ParsePayloadTest, CorruptedPackedWord_ReturnsFailure)
{
    build({make_descriptor(0x61, 0x01, 4)}, {0x01, 0x02, 0x03, 0x04});

    m_packet_buffer[RTP_ST_2110_40_ANCILLARY_HEADER_SIZE + RTP_ST_2110_40_FIXED_DATA_HEADER_SIZE] ^= 0x01;

    RTP_SMPTE_2110_40_PacketContext context;
    RTP_SMPTE_2110_40_PacketReader reader(m_packet_buffer.data(), nullptr);
    ASSERT_EQ(reader.parse_header(context), ReturnStatus::success);
    context.descriptors = &m_out_descriptors;
    context.payload_ptr = m_out_data.data();
    EXPECT_EQ(reader.parse_payload(context), ReturnStatus::failure);
}

/* Second sub-packet corrupt; first preserved in output, descriptor_count_in_packet reflects first only. */
TEST_F(ParsePayloadTest, CorruptSecondSubPacket_FirstSubPacketPreserved)
{
    constexpr uint16_t user_data_words_count_a = 3;
    constexpr uint16_t user_data_words_count_b = 2;

    build({make_descriptor(0x61, 0x01, user_data_words_count_a), make_descriptor(0x62, 0x02, user_data_words_count_b)},
        {0x11, 0x22, 0x33, 0x44, 0x55});

    const size_t sub_pkt_a_size = AncillaryDataUtils::calculate_packet_size(user_data_words_count_a);
    const size_t sub_pkt_b_packed_start =
        RTP_ST_2110_40_ANCILLARY_HEADER_SIZE + sub_pkt_a_size + RTP_ST_2110_40_FIXED_DATA_HEADER_SIZE;
    m_packet_buffer[sub_pkt_b_packed_start] ^= 0x80;

    RTP_SMPTE_2110_40_PacketContext context;
    EXPECT_EQ(parse(context), ReturnStatus::failure);

    ASSERT_EQ(m_out_descriptors.size(), 1u);
    EXPECT_EQ(m_out_descriptors[0].ancillary_data_header.did, 0x61u);
    EXPECT_EQ(m_out_descriptors[0].ancillary_data_header.user_data_words_count, user_data_words_count_a);
    EXPECT_EQ(m_out_descriptors[0].user_data_offset, 0u);

    EXPECT_EQ(context.descriptor_count_in_packet, 1u);

    EXPECT_EQ(m_out_data[0], 0x11u);
    EXPECT_EQ(m_out_data[1], 0x22u);
    EXPECT_EQ(m_out_data[2], 0x33u);
}

/* parse_payload appends to descriptors; two calls without clearing double count. */
TEST_F(ParsePayloadTest, AppendSemantics_NoClear)
{
    build({make_descriptor(0x61, 0x01, 2)}, {0x01, 0x02});

    RTP_SMPTE_2110_40_PacketContext context;
    ASSERT_EQ(parse(context), ReturnStatus::success);
    EXPECT_EQ(m_out_descriptors.size(), 1u);

    RTP_SMPTE_2110_40_PacketReader reader_2(m_packet_buffer.data(), nullptr);
    ASSERT_EQ(reader_2.parse_header(context), ReturnStatus::success);
    ASSERT_EQ(reader_2.parse_payload(context), ReturnStatus::success);
    EXPECT_EQ(m_out_descriptors.size(), 2u);
}

/* descriptor_count_in_packet is per-call only, not cumulative. */
TEST_F(ParsePayloadTest, DescriptorCountInPacket_PerCallOnly)
{
    constexpr uint16_t user_data_words_count = 5;
    build({make_descriptor(0x61, 0x01, user_data_words_count)}, std::vector<uint8_t>(user_data_words_count, 0xAA));

    std::vector<uint8_t> big(user_data_words_count * 2, 0);
    m_out_descriptors.clear();

    RTP_SMPTE_2110_40_PacketContext context;
    context.descriptors = &m_out_descriptors;
    context.payload_ptr = big.data();

    RTP_SMPTE_2110_40_PacketReader reader(m_packet_buffer.data(), nullptr);
    ASSERT_EQ(reader.parse_header(context), ReturnStatus::success);

    ASSERT_EQ(reader.parse_payload(context), ReturnStatus::success);
    EXPECT_EQ(context.descriptor_count_in_packet, 1u);

    context.payload_ptr += user_data_words_count;

    ASSERT_EQ(reader.parse_payload(context), ReturnStatus::success);
    EXPECT_EQ(context.descriptor_count_in_packet, 1u);
}

/* user_data_offset relative to call payload_ptr; two calls verify absolute access. */
TEST_F(ParsePayloadTest, UserDataOffset_RelativeToCallPayloadPtr)
{
    constexpr uint32_t ANCILLARY_COUNT = 3;
    constexpr uint16_t user_data_words_per_packet = 2;

    std::vector<AncillaryDataDescriptor> descriptors;
    std::vector<uint8_t> user_data_words;
    for (uint32_t i = 0; i < ANCILLARY_COUNT; ++i) {
        descriptors.push_back(make_descriptor(
            static_cast<uint8_t>(0x41 + i), 0x01, user_data_words_per_packet,
            0, 0, false, false, 0, static_cast<size_t>(i * user_data_words_per_packet)));
        for (uint16_t j = 0; j < user_data_words_per_packet; ++j) {
            user_data_words.push_back(static_cast<uint8_t>(i * user_data_words_per_packet + j + 1));
        }
    }
    build(descriptors, user_data_words);

    const size_t bytes_per_call = ANCILLARY_COUNT * user_data_words_per_packet;
    std::vector<uint8_t> big(bytes_per_call * 2, 0);
    m_out_descriptors.clear();

    RTP_SMPTE_2110_40_PacketContext context;
    context.descriptors = &m_out_descriptors;
    context.payload_ptr = big.data();

    RTP_SMPTE_2110_40_PacketReader reader(m_packet_buffer.data(), nullptr);
    ASSERT_EQ(reader.parse_header(context), ReturnStatus::success);

    ASSERT_EQ(reader.parse_payload(context), ReturnStatus::success);
    for (uint32_t i = 0; i < ANCILLARY_COUNT; ++i) {
        EXPECT_EQ(m_out_descriptors[i].user_data_offset, i * user_data_words_per_packet) << "call1 i=" << i;
    }

    size_t call1_offset = 0;
    for (size_t i = context.descriptor_start_index; i < context.descriptor_start_index + context.descriptor_count_in_packet; ++i) {
        call1_offset += m_out_descriptors[i].ancillary_data_header.user_data_words_count;
    }
    context.payload_ptr += call1_offset;

    ASSERT_EQ(reader.parse_payload(context), ReturnStatus::success);
    for (uint32_t i = 0; i < ANCILLARY_COUNT; ++i) {
        EXPECT_EQ(m_out_descriptors[ANCILLARY_COUNT + i].user_data_offset, i * user_data_words_per_packet) << "call2 i=" << i;
    }

    for (uint32_t i = 0; i < ANCILLARY_COUNT; ++i) {
        const size_t abs = call1_offset + m_out_descriptors[ANCILLARY_COUNT + i].user_data_offset;
        for (uint16_t j = 0; j < user_data_words_per_packet; ++j) {
            EXPECT_EQ(big[abs + j], user_data_words[i * user_data_words_per_packet + j])
                << "absolute check i=" << i << " j=" << j;
        }
    }
}

/* HDS mode: separate header and payload pointers; output matches non-HDS. */
TEST_F(ParsePayloadTest, HDSMode_SeparateHeaderAndPayload_ProducesCorrectOutput)
{
    constexpr uint16_t user_data_words_count = 4;
    const std::vector<uint8_t> source_user_data_words = {0x11, 0x22, 0x33, 0x44};
    build({make_descriptor(0x61, 0x42, user_data_words_count, 10, 100, true, false, 3)}, source_user_data_words);

    const byte_t* header_ptr = m_packet_buffer.data();
    const byte_t* payload_ptr = m_packet_buffer.data() + RTP_ST_2110_40_ANCILLARY_HEADER_SIZE;

    RTP_SMPTE_2110_40_PacketContext context;
    context.descriptors = &m_out_descriptors;
    context.payload_ptr = m_out_data.data();

    RTP_SMPTE_2110_40_PacketReader reader(header_ptr, payload_ptr);
    ASSERT_EQ(reader.parse_header(context), ReturnStatus::success);
    ASSERT_EQ(reader.parse_payload(context), ReturnStatus::success);

    ASSERT_EQ(m_out_descriptors.size(), 1u);
    EXPECT_EQ(context.descriptor_count_in_packet, 1u);

    const auto& header = m_out_descriptors[0].ancillary_data_header;
    EXPECT_EQ(header.did, 0x61u);
    EXPECT_EQ(header.sdid, 0x42u);
    EXPECT_EQ(header.line_number, 10u);
    EXPECT_EQ(header.horizontal_offset, 100u);
    EXPECT_TRUE(header.c_flag);
    EXPECT_EQ(header.stream_number, 3u);
    for (uint16_t j = 0; j < user_data_words_count; ++j) {
        EXPECT_EQ(m_out_data[j], source_user_data_words[j]) << "j=" << j;
    }
}
