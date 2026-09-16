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

/* Platform specific headers and declarations */
#if defined(_WIN32)
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#elif defined(__linux__)
#include <arpa/inet.h>
#endif

#include <array>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include "rdk/services/media/media_defs.h"
#include "rdk/services/protocol/media_packet_parser.h"
#include "rdk/services/utils/defs.h"

using namespace rdk::services;

namespace
{

/** Offset of the IPv4 header in the test packets. */
constexpr size_t IPV4_OFFSET = sizeof(EthernetHeader);
/** Offset of the UDP header in the test packets. */
constexpr size_t UDP_OFFSET = IPV4_OFFSET + IPV4_MIN_HEADER_SIZE;
/** Offset of the RTP header in the test packets. */
constexpr size_t RTP_OFFSET = UDP_OFFSET + sizeof(UDPHeader);
/** Smallest Ethernet/IPv4/UDP/RTP packet accepted by the media protocol chain. */
constexpr size_t MEDIA_STACK_MIN_PACKET_SIZE = RTP_OFFSET + RTP_HEADER_SIZE;

/** Offset of the UDP datagram length field in the test packets. */
constexpr size_t UDP_LENGTH_OFFSET = UDP_OFFSET + offsetof(UDPHeader, length);

/** Largest number of RTP header bytes that leaves the vpxcc octet outside the packet. */
constexpr size_t RTP_BYTES_WITHOUT_VPXCC = offsetof(protocol::RTPHeader, vpxcc);
/** Largest number of RTP header bytes that leaves the sequence number incomplete. */
constexpr size_t RTP_BYTES_WITH_INCOMPLETE_SEQUENCE_NUMBER =
    offsetof(protocol::RTPHeader, sequence_number) + sizeof(protocol::RTPHeader::sequence_number) - 1;
/** Largest number of RTP bytes that leaves the extended sequence number incomplete. */
constexpr size_t RTP_BYTES_WITH_INCOMPLETE_EXTENDED_SEQUENCE_NUMBER =
    sizeof(protocol::RTPHeader) + RTP_HEADER_EXT_SEQ_NUM_SIZE - 1;

/**
 * @brief: Well-formed Ethernet/IPv4/UDP/RTP packet.
 *
 * The RTP payload starts with an extended sequence number, as in SMPTE ST 2110-20.
 */
constexpr std::array<byte_t, 62> MEDIA_PACKET = {
    /* Ethernet: destination MAC, source MAC, EtherType IPv4 */
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55,
    0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB,
    0x08, 0x00,
    /* IPv4: version 4, IHL 5, total length 48, TTL 64, protocol UDP, 192.168.0.1 -> 239.192.0.1 */
    0x45, 0x00, 0x00, 0x30,
    0x00, 0x00, 0x40, 0x00,
    0x40, 0x11, 0x00, 0x00,
    0xC0, 0xA8, 0x00, 0x01,
    0xEF, 0xC0, 0x00, 0x01,
    /* UDP: ports 20000 -> 50000, length 28, no checksum */
    0x4E, 0x20, 0xC3, 0x50,
    0x00, 0x1C, 0x00, 0x00,
    /* RTP: version 2, no CSRC, payload type 96, sequence number 0x1234 */
    0x80, 0x60, 0x12, 0x34,
    0x11, 0x22, 0x33, 0x44,
    0x0A, 0x0B, 0x0C, 0x0D,
    /* Payload: extended sequence number 0xABCD, followed by filler */
    0xAB, 0xCD, 0x5A, 0x5A, 0x5A, 0x5A, 0x5A, 0x5A
};

/**
 * @brief: Well-formed packet whose RTP header carries a two-entry CSRC list.
 *
 * The extended sequence number follows the CSRC list rather than the fixed RTP header.
 */
constexpr std::array<byte_t, 70> MEDIA_PACKET_WITH_CSRC_LIST = {
    /* Ethernet: destination MAC, source MAC, EtherType IPv4 */
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55,
    0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB,
    0x08, 0x00,
    /* IPv4: version 4, IHL 5, total length 56, TTL 64, protocol UDP, 192.168.0.1 -> 239.192.0.1 */
    0x45, 0x00, 0x00, 0x38,
    0x00, 0x00, 0x40, 0x00,
    0x40, 0x11, 0x00, 0x00,
    0xC0, 0xA8, 0x00, 0x01,
    0xEF, 0xC0, 0x00, 0x01,
    /* UDP: ports 20000 -> 50000, length 36, no checksum */
    0x4E, 0x20, 0xC3, 0x50,
    0x00, 0x24, 0x00, 0x00,
    /* RTP: version 2, two CSRC identifiers, payload type 96, sequence number 0x0001 */
    0x82, 0x60, 0x00, 0x01,
    0x11, 0x22, 0x33, 0x44,
    0x0A, 0x0B, 0x0C, 0x0D,
    /* CSRC list */
    0xDE, 0xAD, 0xBE, 0xEF,
    0xCA, 0xFE, 0xBA, 0xBE,
    /* Payload: extended sequence number 0x0002, followed by filler */
    0x00, 0x02, 0x5A, 0x5A, 0x5A, 0x5A, 0x5A, 0x5A
};

/**
 * @brief: Well-formed packet of the minimum size, leaving no room for an extended sequence number.
 */
constexpr std::array<byte_t, MEDIA_STACK_MIN_PACKET_SIZE> MEDIA_PACKET_WITHOUT_PAYLOAD = {
    /* Ethernet: destination MAC, source MAC, EtherType IPv4 */
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55,
    0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB,
    0x08, 0x00,
    /* IPv4: version 4, IHL 5, total length 40, TTL 64, protocol UDP, 192.168.0.1 -> 239.192.0.1 */
    0x45, 0x00, 0x00, 0x28,
    0x00, 0x00, 0x40, 0x00,
    0x40, 0x11, 0x00, 0x00,
    0xC0, 0xA8, 0x00, 0x01,
    0xEF, 0xC0, 0x00, 0x01,
    /* UDP: ports 20000 -> 50000, length 20, no checksum */
    0x4E, 0x20, 0xC3, 0x50,
    0x00, 0x14, 0x00, 0x00,
    /* RTP: version 2, no CSRC, payload type 96, sequence number 0x1234 */
    0x80, 0x60, 0x12, 0x34,
    0x11, 0x22, 0x33, 0x44,
    0x0A, 0x0B, 0x0C, 0x0D
};

/**
 * @brief: Packet without network headers, as delivered by a receiver configured for header-data split.
 */
constexpr std::array<byte_t, 20> RTP_ONLY_PACKET = {
    /* RTP: version 2, no CSRC, payload type 96, sequence number 0x4321 */
    0x80, 0x60, 0x43, 0x21,
    0x11, 0x22, 0x33, 0x44,
    0x0A, 0x0B, 0x0C, 0x0D,
    /* Payload: extended sequence number 0x0007, followed by filler */
    0x00, 0x07, 0x5A, 0x5A, 0x5A, 0x5A, 0x5A, 0x5A
};

/* Header fields of MEDIA_PACKET. */
constexpr uint8_t PACKET_IP_PROTOCOL_UDP = 17;
constexpr uint32_t PACKET_SRC_IP = 0xC0A80001;
constexpr uint32_t PACKET_DST_IP = 0xEFC00001;
constexpr uint16_t PACKET_SRC_PORT = 20000;
constexpr uint16_t PACKET_DST_PORT = 50000;
constexpr uint32_t PACKET_TIMESTAMP = 0x11223344;
constexpr uint32_t PACKET_SSRC = 0x0A0B0C0D;
constexpr uint32_t PACKET_SEQUENCE_NUMBER = 0x1234;
constexpr uint32_t PACKET_EXTENDED_SEQUENCE_NUMBER = 0xABCD1234;

/* Sequence numbers of MEDIA_PACKET_WITH_CSRC_LIST. */
constexpr uint32_t CSRC_LIST_PACKET_SEQUENCE_NUMBER = 0x0001;
constexpr uint32_t CSRC_LIST_PACKET_EXTENDED_SEQUENCE_NUMBER = 0x00020001;

/* Sequence numbers of RTP_ONLY_PACKET. */
constexpr uint32_t RTP_ONLY_PACKET_SEQUENCE_NUMBER = 0x4321;
constexpr uint32_t RTP_ONLY_PACKET_EXTENDED_SEQUENCE_NUMBER = 0x00074321;

/** Value seeded into the output parameter to detect writes done by a failing extraction. */
constexpr uint32_t SEQUENCE_NUMBER_SENTINEL = 0xDEADBEEF;

/* Malformed field values used by the validation tests. */
constexpr byte_t IPV4_VERSION_IHL_WITH_IP_VERSION_6 = 0x65;
constexpr byte_t RTP_VPXCC_WITH_VERSION_0 = 0x00;
constexpr byte_t RTP_VPXCC_WITH_MAX_CSRC_COUNT = 0x8F;

} // namespace

/**
 * @brief: Test suite for a parser built with the full Ethernet/IPv4/UDP/RTP chain.
 */
class MediaPacketParserTest : public ::testing::Test
{
protected:
    MediaPacketParser m_parser{true};
};

/**
 * @brief: Test suite for a parser built with an RTP-only chain.
 */
class MediaPacketParserRTPOnlyTest : public ::testing::Test
{
protected:
    MediaPacketParser m_parser{false};
};

TEST_F(MediaPacketParserTest, HeaderAccessors_ExposePacketFields)
{
    const byte_t* data = MEDIA_PACKET.data();

    EXPECT_EQ(ntohs(m_parser.ethernet(data)->ether_type), ETH_TYPE_IPV4);
    EXPECT_EQ(m_parser.ipv4(data)->protocol, PACKET_IP_PROTOCOL_UDP);
    EXPECT_EQ(ntohl(m_parser.ipv4(data)->src_ip), PACKET_SRC_IP);
    EXPECT_EQ(ntohl(m_parser.ipv4(data)->dst_ip), PACKET_DST_IP);
    EXPECT_EQ(ntohs(m_parser.udp(data)->src_port), PACKET_SRC_PORT);
    EXPECT_EQ(ntohs(m_parser.udp(data)->dst_port), PACKET_DST_PORT);
    EXPECT_EQ(ntohs(m_parser.rtp(data)->sequence_number), PACKET_SEQUENCE_NUMBER);
    EXPECT_EQ(ntohl(m_parser.rtp(data)->timestamp), PACKET_TIMESTAMP);
    EXPECT_EQ(ntohl(m_parser.rtp(data)->ssrc), PACKET_SSRC);
}

TEST_F(MediaPacketParserTest, GetSequenceNumber_PlainIgnoresExtendedField)
{
    uint32_t sequence_number = 0;

    ASSERT_TRUE(m_parser.get_sequence_number(MEDIA_PACKET.data(), MEDIA_PACKET.size(), false, sequence_number));
    EXPECT_EQ(sequence_number, PACKET_SEQUENCE_NUMBER);
}

TEST_F(MediaPacketParserTest, GetSequenceNumber_ExtendedCombinesBothFields)
{
    uint32_t sequence_number = 0;

    ASSERT_TRUE(m_parser.get_sequence_number(MEDIA_PACKET.data(), MEDIA_PACKET.size(), true, sequence_number));
    EXPECT_EQ(sequence_number, PACKET_EXTENDED_SEQUENCE_NUMBER);
}

TEST_F(MediaPacketParserTest, GetSequenceNumber_ExtendedSkipsCsrcList)
{
    const byte_t* data = MEDIA_PACKET_WITH_CSRC_LIST.data();
    const size_t length = MEDIA_PACKET_WITH_CSRC_LIST.size();
    uint32_t sequence_number = 0;

    ASSERT_TRUE(m_parser.get_sequence_number(data, length, false, sequence_number));
    EXPECT_EQ(sequence_number, CSRC_LIST_PACKET_SEQUENCE_NUMBER);

    ASSERT_TRUE(m_parser.get_sequence_number(data, length, true, sequence_number));
    EXPECT_EQ(sequence_number, CSRC_LIST_PACKET_EXTENDED_SEQUENCE_NUMBER);
}

TEST_F(MediaPacketParserTest, GetSequenceNumber_IncompleteExtendedField_Invalid)
{
    const byte_t* data = MEDIA_PACKET.data();
    uint32_t sequence_number = SEQUENCE_NUMBER_SENTINEL;

    EXPECT_FALSE(m_parser.get_sequence_number(
        data, RTP_OFFSET + RTP_BYTES_WITH_INCOMPLETE_EXTENDED_SEQUENCE_NUMBER, true, sequence_number));
    EXPECT_EQ(sequence_number, SEQUENCE_NUMBER_SENTINEL);

    ASSERT_TRUE(m_parser.get_sequence_number(
        data, RTP_OFFSET + sizeof(protocol::RTPHeader) + RTP_HEADER_EXT_SEQ_NUM_SIZE, true, sequence_number));
    EXPECT_EQ(sequence_number, PACKET_EXTENDED_SEQUENCE_NUMBER);
}

TEST_F(MediaPacketParserTest, GetSequenceNumber_ExtendedFieldBeyondCsrcList_Invalid)
{
    auto packet = MEDIA_PACKET;
    packet[RTP_OFFSET] = RTP_VPXCC_WITH_MAX_CSRC_COUNT;
    uint32_t sequence_number = SEQUENCE_NUMBER_SENTINEL;

    ASSERT_FALSE(m_parser.get_sequence_number(packet.data(), packet.size(), true, sequence_number));
    EXPECT_EQ(sequence_number, SEQUENCE_NUMBER_SENTINEL);
}

TEST_F(MediaPacketParserTest, GetSequenceNumber_PacketShorterThanRtpHeader_Invalid)
{
    const byte_t* data = MEDIA_PACKET.data();
    uint32_t sequence_number = SEQUENCE_NUMBER_SENTINEL;

    EXPECT_FALSE(m_parser.get_sequence_number(data, RTP_OFFSET - 1, true, sequence_number));
    EXPECT_FALSE(m_parser.get_sequence_number(data, RTP_OFFSET + RTP_BYTES_WITHOUT_VPXCC, true, sequence_number));
    EXPECT_FALSE(m_parser.get_sequence_number(data, RTP_OFFSET + RTP_BYTES_WITH_INCOMPLETE_SEQUENCE_NUMBER,
                                              false, sequence_number));
    EXPECT_EQ(sequence_number, SEQUENCE_NUMBER_SENTINEL);
}

TEST_F(MediaPacketParserTest, GetSequenceNumber_NullData_Invalid)
{
    uint32_t sequence_number = SEQUENCE_NUMBER_SENTINEL;

    EXPECT_FALSE(m_parser.get_sequence_number(nullptr, MEDIA_PACKET.size(), false, sequence_number));
    EXPECT_FALSE(m_parser.get_sequence_number(nullptr, MEDIA_PACKET.size(), true, sequence_number));
    EXPECT_EQ(sequence_number, SEQUENCE_NUMBER_SENTINEL);
}

TEST_F(MediaPacketParserTest, GetSequenceNumber_FailedExtendedReadKeepsEarlierResult)
{
    const byte_t* data = MEDIA_PACKET.data();
    uint32_t sequence_number = 0;

    ASSERT_TRUE(m_parser.get_sequence_number(data, MEDIA_PACKET.size(), false, sequence_number));
    ASSERT_EQ(sequence_number, PACKET_SEQUENCE_NUMBER);

    ASSERT_FALSE(m_parser.get_sequence_number(
        data, RTP_OFFSET + RTP_BYTES_WITH_INCOMPLETE_EXTENDED_SEQUENCE_NUMBER, true, sequence_number));
    EXPECT_EQ(sequence_number, PACKET_SEQUENCE_NUMBER);
}

TEST_F(MediaPacketParserTest, Validate_WellFormedPacketsAreAccepted)
{
    EXPECT_TRUE(m_parser.validate(MEDIA_PACKET.data(), MEDIA_PACKET.size()));
    EXPECT_TRUE(m_parser.validate(MEDIA_PACKET_WITH_CSRC_LIST.data(), MEDIA_PACKET_WITH_CSRC_LIST.size()));
    EXPECT_TRUE(m_parser.validate(MEDIA_PACKET_WITHOUT_PAYLOAD.data(), MEDIA_PACKET_WITHOUT_PAYLOAD.size()));
}

TEST_F(MediaPacketParserTest, Validate_PacketBelowMinimumSizeIsRejected)
{
    EXPECT_FALSE(m_parser.validate(MEDIA_PACKET.data(), MEDIA_STACK_MIN_PACKET_SIZE - 1));
}

TEST_F(MediaPacketParserTest, Validate_UnexpectedIpVersionIsRejected)
{
    auto packet = MEDIA_PACKET;
    packet[IPV4_OFFSET] = IPV4_VERSION_IHL_WITH_IP_VERSION_6;

    EXPECT_FALSE(m_parser.validate(packet.data(), packet.size()));
}

TEST_F(MediaPacketParserTest, Validate_UdpLengthBeyondPacketEndIsRejected)
{
    auto packet = MEDIA_PACKET;
    packet[UDP_LENGTH_OFFSET] = 0xFF;
    packet[UDP_LENGTH_OFFSET + 1] = 0xFF;

    EXPECT_FALSE(m_parser.validate(packet.data(), packet.size()));
}

TEST_F(MediaPacketParserTest, Validate_UnexpectedRtpVersionIsRejected)
{
    auto packet = MEDIA_PACKET;
    packet[RTP_OFFSET] = RTP_VPXCC_WITH_VERSION_0;

    EXPECT_FALSE(m_parser.validate(packet.data(), packet.size()));
}

TEST_F(MediaPacketParserTest, Validate_CsrcListBeyondPacketEndIsRejected)
{
    auto packet = MEDIA_PACKET;
    packet[RTP_OFFSET] = RTP_VPXCC_WITH_MAX_CSRC_COUNT;

    EXPECT_FALSE(m_parser.validate(packet.data(), packet.size()));
}

TEST_F(MediaPacketParserRTPOnlyTest, ProtocolChain_ContainsRtpLayerOnly)
{
    ASSERT_EQ(m_parser.get_parser_count(), 1u);
    EXPECT_EQ(m_parser.get_min_packet_size(), RTP_HEADER_SIZE);
    EXPECT_TRUE(m_parser.supports_header_type<protocol::RTPHeader>());
    EXPECT_FALSE(m_parser.supports_header_type<EthernetHeader>());
    EXPECT_FALSE(m_parser.supports_header_type<IPv4Header>());
    EXPECT_FALSE(m_parser.supports_header_type<UDPHeader>());
}

TEST_F(MediaPacketParserRTPOnlyTest, HeaderAccessors_OnlyRtpIsAvailable)
{
    const byte_t* data = RTP_ONLY_PACKET.data();

    EXPECT_EQ(reinterpret_cast<const byte_t*>(m_parser.rtp(data)), data);
    EXPECT_EQ(m_parser.ethernet(data), nullptr);
    EXPECT_EQ(m_parser.ipv4(data), nullptr);
    EXPECT_EQ(m_parser.udp(data), nullptr);
}

TEST_F(MediaPacketParserRTPOnlyTest, GetSequenceNumber_ReadsFieldsAtPacketStart)
{
    const byte_t* data = RTP_ONLY_PACKET.data();
    const size_t length = RTP_ONLY_PACKET.size();
    uint32_t sequence_number = 0;

    ASSERT_TRUE(m_parser.get_sequence_number(data, length, false, sequence_number));
    EXPECT_EQ(sequence_number, RTP_ONLY_PACKET_SEQUENCE_NUMBER);

    ASSERT_TRUE(m_parser.get_sequence_number(data, length, true, sequence_number));
    EXPECT_EQ(sequence_number, RTP_ONLY_PACKET_EXTENDED_SEQUENCE_NUMBER);
}

TEST_F(MediaPacketParserRTPOnlyTest, GetSequenceNumber_PacketShorterThanRtpHeader_Invalid)
{
    const byte_t* data = RTP_ONLY_PACKET.data();
    uint32_t sequence_number = SEQUENCE_NUMBER_SENTINEL;

    EXPECT_FALSE(m_parser.get_sequence_number(data, RTP_BYTES_WITHOUT_VPXCC, true, sequence_number));
    EXPECT_FALSE(m_parser.get_sequence_number(data, RTP_BYTES_WITH_INCOMPLETE_SEQUENCE_NUMBER,
                                              false, sequence_number));
    EXPECT_EQ(sequence_number, SEQUENCE_NUMBER_SENTINEL);

    EXPECT_TRUE(m_parser.get_sequence_number(data, RTP_HEADER_SIZE, false, sequence_number));
    EXPECT_EQ(sequence_number, RTP_ONLY_PACKET_SEQUENCE_NUMBER);
}

TEST_F(MediaPacketParserRTPOnlyTest, GetSequenceNumber_IncompleteExtendedField_Invalid)
{
    const byte_t* data = RTP_ONLY_PACKET.data();
    uint32_t sequence_number = SEQUENCE_NUMBER_SENTINEL;

    EXPECT_FALSE(m_parser.get_sequence_number(data, RTP_BYTES_WITH_INCOMPLETE_EXTENDED_SEQUENCE_NUMBER,
                                              true, sequence_number));
    EXPECT_EQ(sequence_number, SEQUENCE_NUMBER_SENTINEL);

    ASSERT_TRUE(m_parser.get_sequence_number(data, sizeof(protocol::RTPHeader) + RTP_HEADER_EXT_SEQ_NUM_SIZE,
                                             true, sequence_number));
    EXPECT_EQ(sequence_number, RTP_ONLY_PACKET_EXTENDED_SEQUENCE_NUMBER);
}

TEST_F(MediaPacketParserRTPOnlyTest, GetSequenceNumber_NullData_Invalid)
{
    uint32_t sequence_number = SEQUENCE_NUMBER_SENTINEL;

    EXPECT_FALSE(m_parser.get_sequence_number(nullptr, RTP_ONLY_PACKET.size(), false, sequence_number));
    EXPECT_FALSE(m_parser.get_sequence_number(nullptr, RTP_ONLY_PACKET.size(), true, sequence_number));
    EXPECT_EQ(sequence_number, SEQUENCE_NUMBER_SENTINEL);
}
