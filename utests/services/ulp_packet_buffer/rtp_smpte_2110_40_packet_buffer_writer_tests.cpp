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

#include <memory>

#include <gtest/gtest.h>

#include "rdk/services/ulp_packet_buffer/writers/rtp_smpte_2110_40_packet_buffer_writer.h"
#include "rdk/services/media/media_essence_source.h"
#include "rdk/services/media/media_defs.h"
#include "rdk/services/media/ancillary_metadata.h"
#include "rdk/services/media/ancillary_settings_calculator.h"
#include "rdk/services/memory_allocation/new_memory_allocator.h"
#include "rdk/services/utils/rational.h"

using namespace rdk::services;
using namespace testing;
using rdk::services::rational_cast;

struct TestRTPHeader {
    uint8_t cc : 4;
    uint8_t extension : 1;
    uint8_t padding : 1;
    uint8_t version : 2;
    uint8_t payload_type : 7;
    uint8_t marker : 1;
    uint16_t sequence_number;
    uint32_t timestamp;
    uint32_t ssrc;
};

struct TestAncillaryRTPExtension {
    uint16_t extended_sequence_number;
    uint16_t length;
    uint8_t anc_count;
    uint8_t reserved_byte1;
    uint16_t reserved_bytes2_3;
};

static std::shared_ptr<SMPTE_2110_40_MediaSettings> create_basic_settings(
    uint32_t packets_in_media_unit, size_t chunks_in_media_unit, uint16_t max_user_data_words,
    uint32_t max_ancillary_data_packets_per_packet, bool enable_hds)
{
    auto settings = std::make_shared<SMPTE_2110_40_MediaSettings>();
    settings->max_user_data_words_count = max_user_data_words;
    settings->max_ancillary_data_packets_per_packet = max_ancillary_data_packets_per_packet;
    settings->data_identifiers = {{0x41, 0x01}};
    settings->packets_in_media_unit = packets_in_media_unit;
    settings->chunks_in_media_unit = chunks_in_media_unit;
    settings->header_data_split = enable_hds;

    ReturnStatus status = settings->create_default_calculator();
    if (status != ReturnStatus::success) {
        ADD_FAILURE() << "Failed to create default calculator for 2110-40 settings";
        return nullptr;
    }
    return settings;
}

static std::shared_ptr<MediaUnit> create_ancillary_media_unit(
    const std::shared_ptr<SMPTE_2110_40_MediaSettings>& media_settings,
    size_t ancillary_data_packets_count, uint16_t words_per_ancillary_data_packet)
{
    auto media_unit = std::make_shared<MediaUnit>(media_settings->bytes_per_media_unit,
                                                  SMPTEStandard::ST_2110_40);
    auto ancillary_metadata =
        std::dynamic_pointer_cast<AncillaryMediaUnitMetadata>(media_unit->metadata);
    if (!ancillary_metadata) {
        return nullptr;
    }

    constexpr uint16_t line_number_value = 10;
    constexpr uint16_t horizontal_offset_value = 100;
    uint8_t* buffer = reinterpret_cast<uint8_t*>(media_unit->data->get());
    size_t cumulative_offset = 0;
    uint8_t test_value = 0;

    for (size_t i = 0; i < ancillary_data_packets_count; ++i) {
        AncillaryDataDescriptor descriptor{};
        descriptor.ancillary_data_header.did = media_settings->data_identifiers[0].did;
        descriptor.ancillary_data_header.sdid = media_settings->data_identifiers[0].sdid;
        descriptor.ancillary_data_header.user_data_words_count = words_per_ancillary_data_packet;
        descriptor.ancillary_data_header.line_number = line_number_value;
        descriptor.ancillary_data_header.horizontal_offset = horizontal_offset_value;
        descriptor.user_data_offset = cumulative_offset;

        // Fill with known incrementing pattern
        for (uint16_t j = 0; j < words_per_ancillary_data_packet; ++j) {
            buffer[cumulative_offset + j] = test_value;
            test_value = (test_value + 1) % 10;
        }

        cumulative_offset += words_per_ancillary_data_packet;
        ancillary_metadata->ancillary_data.push_back(descriptor);
    }

    return media_unit;
}

/** Return RTP header view at packet start. */
static const TestRTPHeader* as_rtp_header(const uint8_t* packet)
{
    return reinterpret_cast<const TestRTPHeader*>(packet);
}

/** Return ancillary extension view after RTP header. */
static const TestAncillaryRTPExtension* as_anc_ext(const uint8_t* packet)
{
    return reinterpret_cast<const TestAncillaryRTPExtension*>(packet + sizeof(TestRTPHeader));
}

static uint8_t extract_f_bit(const uint8_t* rtp_packet)
{
    const TestAncillaryRTPExtension* anc_ext = as_anc_ext(rtp_packet);
    // F bits are in bits 6-7 of reserved_byte1
    return (anc_ext->reserved_byte1 >> 6) & 0x03;
}

static uint32_t extract_timestamp(const uint8_t* rtp_packet)
{
    return ntohl(as_rtp_header(rtp_packet)->timestamp);
}

static uint8_t extract_marker_bit(const uint8_t* rtp_packet)
{
    return as_rtp_header(rtp_packet)->marker;
}

/**
 * @brief: SMPTE 2110-40 buffer writer test suite.
 *
 * Tests cover various scenarios of ancillary data packing into RTP packets.
 * Actual user data payload is not tested.
 */
class RTP_SMPTE_2110_40_PacketBufferWriterTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_mem_allocator = std::make_shared<NewMemoryAllocator>();
        m_header_mem_utils = m_mem_allocator->get_memory_utils();
        m_payload_mem_utils = m_mem_allocator->get_memory_utils();
    }

    /** Create media settings and store in m_settings. */
    void init_settings(uint32_t packets_in_media_unit, size_t chunks_in_media_unit,
                       uint16_t max_user_data_words = 255,
                       uint32_t max_anc_per_packet = 10,
                       bool enable_hds = false)
    {
        m_settings = create_basic_settings(packets_in_media_unit, chunks_in_media_unit,
                                           max_user_data_words, max_anc_per_packet, enable_hds);
        ASSERT_NE(m_settings, nullptr);
    }

    /** Configure scan type on m_settings and recalculate. */
    void configure_scan_type(VideoScanType scan_type, FrameRate frame_rate = FrameRate(25))
    {
        m_settings->video_scan_type = scan_type;
        m_settings->frame_rate = frame_rate;
        ST_2110_40_MediaSettingsCalculator calculator(*m_settings);
        ASSERT_EQ(calculator.calculate_media_settings(), ReturnStatus::success);
    }

    /** Create the packet buffer writer and store in m_writer. */
    void init_writer(bool enable_zero_copy = false)
    {
        m_writer = std::make_unique<RTP_SMPTE_2110_40_PacketBufferWriter>(
            *m_settings, m_header_mem_utils, m_payload_mem_utils, enable_zero_copy);
    }

    /** Allocate scratch buffer and payload sizes for the given number of packets. */
    void allocate_buffers(size_t num_packets)
    {
        m_scratch_buffer.resize(m_settings->data_stride_size * num_packets);
        m_payload_sizes.resize(num_packets);
    }

    /** Allocate separate header and payload buffers for HDS mode. */
    void allocate_hds_buffers(size_t num_packets)
    {
        m_header_buffer.resize(m_settings->app_header_stride_size * num_packets);
        m_scratch_buffer.resize(m_settings->data_stride_size * num_packets);
        m_header_sizes.resize(num_packets, 0);
        m_payload_sizes.resize(num_packets, 0);
    }

    /** Create an ancillary media unit and set it on the writer. */
    void set_media_unit(size_t anc_count, uint16_t words_per_packet)
    {
        auto media_unit = create_ancillary_media_unit(m_settings, anc_count, words_per_packet);
        ASSERT_NE(media_unit, nullptr);
        m_writer->set_next_media_unit(std::move(media_unit));
    }

    /** Write a chunk of packets into m_scratch_buffer/m_payload_sizes. */
    ReturnStatus write_chunk(size_t num_packets)
    {
        return m_writer->write_buffer(m_scratch_buffer.data(), num_packets, m_payload_sizes.data());
    }

    /** Write a chunk in HDS mode into m_header_buffer + m_scratch_buffer. */
    ReturnStatus write_hds_chunk(size_t num_packets)
    {
        return m_writer->write_buffer(m_header_buffer.data(), m_scratch_buffer.data(),
                                      num_packets, m_header_sizes.data(), m_payload_sizes.data());
    }

    /** Return pointer to raw packet data at the given index in m_scratch_buffer. */
    uint8_t* packet_data_at(size_t packet_idx)
    {
        return m_scratch_buffer.data() + (packet_idx * m_settings->data_stride_size);
    }

    /** Return typed RTP header pointer at the given packet index. */
    TestRTPHeader* rtp_header_at(size_t packet_idx)
    {
        return reinterpret_cast<TestRTPHeader*>(packet_data_at(packet_idx));
    }

    /** Return typed ancillary extension pointer at the given packet index. */
    TestAncillaryRTPExtension* anc_ext_at(size_t packet_idx)
    {
        return reinterpret_cast<TestAncillaryRTPExtension*>(
            packet_data_at(packet_idx) + sizeof(TestRTPHeader));
    }

    /** Return typed RTP header pointer at the given packet index in HDS header buffer. */
    TestRTPHeader* hds_rtp_header_at(size_t packet_idx)
    {
        uint8_t* ptr = m_header_buffer.data() + (packet_idx * m_settings->app_header_stride_size);
        return reinterpret_cast<TestRTPHeader*>(ptr);
    }

    /** Return typed ancillary extension at the given packet index in HDS header buffer. */
    TestAncillaryRTPExtension* hds_anc_ext_at(size_t packet_idx)
    {
        uint8_t* ptr = m_header_buffer.data() + (packet_idx * m_settings->app_header_stride_size);
        return reinterpret_cast<TestAncillaryRTPExtension*>(ptr + sizeof(TestRTPHeader));
    }

    std::shared_ptr<NewMemoryAllocator> m_mem_allocator;
    std::shared_ptr<MemoryUtils> m_header_mem_utils;
    std::shared_ptr<MemoryUtils> m_payload_mem_utils;
    std::shared_ptr<SMPTE_2110_40_MediaSettings> m_settings;
    std::unique_ptr<RTP_SMPTE_2110_40_PacketBufferWriter> m_writer;
    std::vector<uint8_t> m_scratch_buffer;
    std::vector<uint16_t> m_payload_sizes;
    std::vector<uint8_t> m_header_buffer;
    std::vector<uint16_t> m_header_sizes;
};

/* Test: Mock mode returns fixed packets_in_chunk for all chunks */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, MockMode_FixedPacketsPerChunk)
{
    constexpr size_t chunks = 3;
    constexpr size_t expected_packets_per_chunk = 2;  // 6 / 3

    init_settings(6, chunks);
    init_writer(true);
    allocate_buffers(expected_packets_per_chunk);

    for (size_t chunk_idx = 0; chunk_idx < chunks; ++chunk_idx) {
        EXPECT_EQ(m_writer->get_num_packets_for_next_chunk(), expected_packets_per_chunk)
            << "Chunk " << chunk_idx << " should return fixed packets_per_chunk in zero copy mode";

        EXPECT_EQ(write_chunk(expected_packets_per_chunk), ReturnStatus::success)
            << "Chunk " << chunk_idx << " write_buffer failed";

        // Verify marker bit is only set on the last packet of the media unit
        for (size_t packet_idx = 0; packet_idx < expected_packets_per_chunk; ++packet_idx) {
            bool is_last_packet_of_media_unit = (chunk_idx == chunks - 1) &&
                (packet_idx == expected_packets_per_chunk - 1);

            if (is_last_packet_of_media_unit) {
                EXPECT_EQ(rtp_header_at(packet_idx)->marker, 1)
                    << "Last packet (chunk " << chunk_idx << ", packet " << packet_idx
                    << ") should have marker bit set";
            } else {
                EXPECT_EQ(rtp_header_at(packet_idx)->marker, 0)
                    << "Non-last packet (chunk " << chunk_idx << ", packet " << packet_idx
                    << ") should NOT have marker bit set";
            }
        }
    }
}

/* Test: 3 small ancillary descriptors packed into 1 RTP packet, 1 chunk */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, ThreeAncillaryDescriptors_OneChunk)
{
    init_settings(10, 1);
    init_writer();
    set_media_unit(3, 10);
    allocate_buffers(m_settings->packets_in_media_unit);

    EXPECT_EQ(m_writer->get_num_packets_for_next_chunk(), 1);
    EXPECT_EQ(write_chunk(1), ReturnStatus::success);
    EXPECT_GT(m_payload_sizes[0], 0);

    EXPECT_EQ(anc_ext_at(0)->anc_count, 3);
    EXPECT_EQ(rtp_header_at(0)->marker, 1);
}

/* Test: 3 small ancillary descriptors packed into 1 RTP packet, 2 chunks */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, ThreeAncillaryDescriptors_TwoChunks)
{
    init_settings(10, 2);
    init_writer();
    set_media_unit(3, 10);
    allocate_buffers(m_settings->packets_in_media_unit / m_settings->chunks_in_media_unit);

    //   Chunk 1: 1 packet with 3 ancillary descriptors
    //   Chunk 2: 1 packet with 0 ancillary descriptors and marker bit set
    EXPECT_EQ(m_writer->get_num_packets_for_next_chunk(), 1);
    EXPECT_EQ(write_chunk(1), ReturnStatus::success);
    EXPECT_GT(m_payload_sizes[0], 0);

    EXPECT_EQ(anc_ext_at(0)->anc_count, 3);
    EXPECT_EQ(rtp_header_at(0)->marker, 0);

    // Second chunk should have 1 packet with 0 data and Mark bit set
    EXPECT_GE(m_writer->get_num_packets_for_next_chunk(), 1);
    EXPECT_EQ(write_chunk(1), ReturnStatus::success);

    EXPECT_EQ(anc_ext_at(0)->anc_count, 0);
    EXPECT_EQ(rtp_header_at(0)->marker, 1);
}

/* Test: 3 small ancillary descriptors packed into 2 RTP packets, 1 chunk */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, ThreeAncillaryDescriptors_TwoRTPPackets_OneChunk)
{
    init_settings(10, 1, 255, 2);
    init_writer();
    set_media_unit(3, 10);
    allocate_buffers(m_settings->packets_in_media_unit);

    // 3 ancillary descriptors, max 2 per RTP packet -> packs into 2 RTP packets
    EXPECT_EQ(m_writer->get_num_packets_for_next_chunk(), 2);
    EXPECT_EQ(write_chunk(2), ReturnStatus::success);
    EXPECT_GT(m_payload_sizes[0], 0);
    EXPECT_GT(m_payload_sizes[1], 0);

    // First packet: 2 descriptors, Mark bit NOT set
    EXPECT_EQ(anc_ext_at(0)->anc_count, 2);
    EXPECT_EQ(rtp_header_at(0)->marker, 0);

    // Second packet: 1 descriptor, Mark bit set
    EXPECT_EQ(anc_ext_at(1)->anc_count, 1);
    EXPECT_EQ(rtp_header_at(1)->marker, 1);
}

/* Test: 3 small ancillary descriptors packed into 2 RTP packets, too many chunks (3) */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest,
       ThreeAncillaryDescriptors_TwoRTPPackets_TooManyChunks)
{
    init_settings(10, 3, 255, 2);
    init_writer();
    set_media_unit(3, 10);
    allocate_buffers(m_settings->packets_in_media_unit / m_settings->chunks_in_media_unit);

    // 3 ancillary descriptors, max 2 per RTP packet -> packs into 2 RTP packets
    // With 3 chunks:
    //   Chunk 1: 2 RTP packets with Mark bit not set
    //   Chunk 2: 1 empty RTP packet with Mark bit not set
    //   Chunk 3: 1 empty RTP packet with Mark bit set
    EXPECT_EQ(m_writer->get_num_packets_for_next_chunk(), 2);
    EXPECT_EQ(write_chunk(2), ReturnStatus::success);
    EXPECT_GT(m_payload_sizes[0], 0);
    EXPECT_GT(m_payload_sizes[1], 0);

    // Chunk 1: First packet: 2 descriptors, Mark bit NOT set
    EXPECT_EQ(anc_ext_at(0)->anc_count, 2);
    EXPECT_EQ(rtp_header_at(0)->marker, 0);

    // Chunk 1: Second packet: 1 descriptor, Mark bit NOT set
    EXPECT_EQ(anc_ext_at(1)->anc_count, 1);
    EXPECT_EQ(rtp_header_at(1)->marker, 0);

    // Chunk 2: 0 descriptors, Mark bit NOT set
    EXPECT_EQ(m_writer->get_num_packets_for_next_chunk(), 1);
    EXPECT_EQ(write_chunk(1), ReturnStatus::success);

    EXPECT_EQ(anc_ext_at(0)->anc_count, 0);
    EXPECT_EQ(rtp_header_at(0)->marker, 0);

    // Chunk 3: 0 descriptors, Mark bit set
    EXPECT_EQ(m_writer->get_num_packets_for_next_chunk(), 1);
    EXPECT_EQ(write_chunk(1), ReturnStatus::success);

    EXPECT_EQ(anc_ext_at(0)->anc_count, 0);
    EXPECT_EQ(rtp_header_at(0)->marker, 1);
}

/* Test: 10 ancillary descriptors with 50 user data words - all fit in 1 RTP packet */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, TenANC_50Words_FitsInOneRTPPacket)
{
    init_settings(10, 1);
    init_writer();
    set_media_unit(10, 50);

    constexpr size_t expected_rtp_packets = 1;
    EXPECT_EQ(m_writer->get_num_packets_for_next_chunk(), expected_rtp_packets);
    allocate_buffers(expected_rtp_packets);

    EXPECT_EQ(write_chunk(expected_rtp_packets), ReturnStatus::success);
    EXPECT_GT(m_payload_sizes[0], 0);

    EXPECT_EQ(anc_ext_at(0)->anc_count, 10)
        << "Single packet should contain all 10 ancillary descriptors";
    EXPECT_EQ(rtp_header_at(0)->marker, 1) << "Single packet should have marker bit set";
}

/* Test: 10 ANC with 255 user data words - maximum size constraint */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, TenANC_255Words_MaxPayloadSizeConstrained)
{
    init_settings(10, 1);
    init_writer();
    set_media_unit(10, 255);

    // 10 ancillary descriptors with 255 user data words each
    // Expects 3 full RTP packets to pack all user data words
    constexpr size_t expected_rtp_packets = 3;
    EXPECT_EQ(m_writer->get_num_packets_for_next_chunk(), expected_rtp_packets);
    allocate_buffers(expected_rtp_packets);

    EXPECT_EQ(write_chunk(expected_rtp_packets), ReturnStatus::success);

    for (size_t i = 0; i < expected_rtp_packets; ++i) {
        EXPECT_GT(m_payload_sizes[i], 0) << "Packet " << i << " should have non-zero payload";
    }

    EXPECT_EQ(rtp_header_at(expected_rtp_packets - 1)->marker, 1)
        << "Last packet should have marker bit set";

    for (size_t i = 0; i < expected_rtp_packets; ++i) {
        EXPECT_LE(m_payload_sizes[i], m_settings->packet_payload_size)
            << "Packet " << i << " exceeds max payload size";
    }
}

/* Test: 10 ancillary descriptors with 50 user data words - all fit in 1 RTP packet - HDS mode */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, TenANC_50Words_FitsInOneRTPPacket_HDS)
{
    init_settings(10, 1, 255, 10, true);
    init_writer();
    set_media_unit(10, 50);

    constexpr size_t expected_rtp_packets = 1;
    EXPECT_EQ(m_writer->get_num_packets_for_next_chunk(), expected_rtp_packets);
    allocate_hds_buffers(expected_rtp_packets);

    EXPECT_EQ(write_hds_chunk(expected_rtp_packets), ReturnStatus::success);
    EXPECT_GT(m_header_sizes[0], 0);
    EXPECT_GT(m_payload_sizes[0], 0);

    EXPECT_EQ(hds_anc_ext_at(0)->anc_count, 10)
        << "Single packet should contain all 10 ancillary descriptors";
    EXPECT_EQ(hds_rtp_header_at(0)->marker, 1) << "Single packet should have marker bit set";
}

/* Test: 10 ANC with 255 user data words - maximum size constraint - HDS mode */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, TenANC_255Words_MaxPayloadSizeConstrained_HDS)
{
    init_settings(10, 1, 255, 10, true);
    init_writer();
    set_media_unit(10, 255);

    // 10 ancillary descriptors with 255 user data words each
    // Expects 3 full RTP packets to pack all user data words
    constexpr size_t expected_rtp_packets = 3;
    EXPECT_EQ(m_writer->get_num_packets_for_next_chunk(), expected_rtp_packets);
    allocate_hds_buffers(expected_rtp_packets);

    EXPECT_EQ(write_hds_chunk(expected_rtp_packets), ReturnStatus::success);

    for (size_t i = 0; i < expected_rtp_packets; ++i) {
        EXPECT_GT(m_header_sizes[i], 0) << "Packet " << i << " should have non-zero header";
        EXPECT_GT(m_payload_sizes[i], 0) << "Packet " << i << " should have non-zero payload";
    }

    EXPECT_EQ(hds_rtp_header_at(expected_rtp_packets - 1)->marker, 1)
        << "Last packet should have marker bit set";

    for (size_t i = 0; i < expected_rtp_packets; ++i) {
        EXPECT_LE(m_payload_sizes[i], m_settings->packet_payload_size)
            << "Packet " << i << " exceeds max payload size";
    }
}

/* Test: Progressive scan has F-bit=0b00 */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, ProgressiveScan_FBitZero)
{
    init_settings(2, 1, 128);
    configure_scan_type(VideoScanType::Progressive);
    init_writer();
    set_media_unit(2, 10);
    allocate_buffers(m_settings->packets_in_media_unit);

    size_t num_packets = m_writer->get_num_packets_for_next_chunk();
    EXPECT_EQ(write_chunk(num_packets), ReturnStatus::success);

    // Progressive has F=0b00 (0)
    for (size_t i = 0; i < num_packets; ++i) {
        EXPECT_EQ(extract_f_bit(packet_data_at(i)), 0)
            << "Progressive packet " << i << " should have F=0b00";
    }
}

/* Test: Interlaced scan starts with F-bit=0b10 (first field) */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, InterlacedScan_FirstFieldFBit0b10)
{
    init_settings(2, 1, 128);
    configure_scan_type(VideoScanType::Interlaced);
    init_writer();
    set_media_unit(2, 10);
    allocate_buffers(m_settings->packets_in_media_unit);

    size_t num_packets = m_writer->get_num_packets_for_next_chunk();
    EXPECT_EQ(write_chunk(num_packets), ReturnStatus::success);

    // First field has F=0b10 (2)
    for (size_t i = 0; i < num_packets; ++i) {
        EXPECT_EQ(extract_f_bit(packet_data_at(i)), 2)
            << "First field packet " << i << " should have F=0b10";
    }
}

/* Test: Interlaced scan F-bit alternates 0b10->0b11->0b10->0b11 */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, InterlacedScan_FBitAlternatesBetweenFields)
{
    init_settings(2, 1, 128);
    configure_scan_type(VideoScanType::Interlaced);
    init_writer();
    allocate_buffers(m_settings->packets_in_media_unit);

    // Field 1=0b10 (2), Field 2=0b11 (3), repeating
    const uint8_t expected_f_bits[4] = {2, 3, 2, 3};

    for (int field = 0; field < 4; ++field) {
        set_media_unit(2, 10);

        size_t num_packets = m_writer->get_num_packets_for_next_chunk();
        EXPECT_EQ(write_chunk(num_packets), ReturnStatus::success);

        for (size_t i = 0; i < num_packets; ++i) {
            EXPECT_EQ(extract_f_bit(packet_data_at(i)), expected_f_bits[field])
                << "Field " << field << " packet " << i << " F-bit mismatch";
        }
    }
}

/* Test: RTP timestamp increments per field for interlaced */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, InterlacedMode_TimestampIncrementsPerField)
{
    init_settings(2, 1, 128);
    configure_scan_type(VideoScanType::Interlaced);
    init_writer();
    allocate_buffers(m_settings->packets_in_media_unit);

    uint32_t field1_timestamp = 0;
    uint32_t field2_timestamp = 0;
    uint32_t field3_timestamp = 0;

    // Field 1
    set_media_unit(2, 10);
    size_t num_packets = m_writer->get_num_packets_for_next_chunk();
    EXPECT_EQ(write_chunk(num_packets), ReturnStatus::success);

    field1_timestamp = extract_timestamp(packet_data_at(0));
    for (size_t i = 0; i < num_packets; ++i) {
        EXPECT_EQ(extract_timestamp(packet_data_at(i)), field1_timestamp)
            << "All packets in field 1 should have same timestamp";
    }

    // Field 2
    set_media_unit(2, 10);
    num_packets = m_writer->get_num_packets_for_next_chunk();
    EXPECT_EQ(write_chunk(num_packets), ReturnStatus::success);

    field2_timestamp = extract_timestamp(packet_data_at(0));
    for (size_t i = 0; i < num_packets; ++i) {
        EXPECT_EQ(extract_timestamp(packet_data_at(i)), field2_timestamp)
            << "All packets in field 2 should have same timestamp";
    }

    // Verify timestamp increment
    EXPECT_EQ(field2_timestamp - field1_timestamp,
              static_cast<uint32_t>(m_settings->ticks_per_media_unit))
        << "Timestamp should increment by ticks_per_media_unit";

    // Field 3 - verify consistent increment
    set_media_unit(2, 10);
    num_packets = m_writer->get_num_packets_for_next_chunk();
    EXPECT_EQ(write_chunk(num_packets), ReturnStatus::success);

    field3_timestamp = extract_timestamp(packet_data_at(0));
    EXPECT_EQ(field3_timestamp - field2_timestamp,
              static_cast<uint32_t>(m_settings->ticks_per_media_unit))
        << "Timestamp increment should be consistent between fields";
}

/* Test: Marker bit set on last packet of field */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, InterlacedMode_MarkerBitOnLastPacketOfField)
{
    init_settings(2, 1, 128);
    configure_scan_type(VideoScanType::Interlaced);
    init_writer();
    allocate_buffers(m_settings->packets_in_media_unit);

    // Test multiple fields
    for (int field = 0; field < 3; ++field) {
        set_media_unit(2, 10);

        size_t num_packets = m_writer->get_num_packets_for_next_chunk();
        EXPECT_EQ(write_chunk(num_packets), ReturnStatus::success);

        // Verify marker bit is set only on last packet
        for (size_t i = 0; i < num_packets; ++i) {
            uint8_t marker = extract_marker_bit(packet_data_at(i));

            if (i == num_packets - 1) {
                EXPECT_EQ(marker, 1)
                    << "Field " << field << ": Last packet should have marker bit set";
            } else {
                EXPECT_EQ(marker, 0)
                    << "Field " << field << ": Non-last packet should not have marker bit set";
            }
        }
    }
}

/* Test: F-bit consistency across all packets within same field */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, InterlacedMode_FBitConsistentWithinField)
{
    init_settings(5, 1, 128);
    configure_scan_type(VideoScanType::Interlaced);
    init_writer();
    allocate_buffers(m_settings->packets_in_media_unit);

    // Test 4 fields
    for (int field = 0; field < 4; ++field) {
        set_media_unit(5, 50);

        uint8_t expected_f_bit = (field % 2 == 0) ? 2 : 3;

        size_t num_packets = m_writer->get_num_packets_for_next_chunk();
        EXPECT_EQ(write_chunk(num_packets), ReturnStatus::success);

        // Verify ALL packets in this field have the same F-bit
        for (size_t i = 0; i < num_packets; ++i) {
            EXPECT_EQ(extract_f_bit(packet_data_at(i)), expected_f_bit)
                << "Field " << field << " packet " << i << " F-bit mismatch";
        }
    }
}

/* Test: Timestamp consistency across all packets within same field */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, InterlacedMode_TimestampConsistentWithinField)
{
    init_settings(5, 1, 128);
    configure_scan_type(VideoScanType::Interlaced);
    init_writer();
    set_media_unit(5, 50);
    allocate_buffers(m_settings->packets_in_media_unit);

    size_t num_packets = m_writer->get_num_packets_for_next_chunk();
    EXPECT_EQ(write_chunk(num_packets), ReturnStatus::success);

    // Get first packet timestamp
    uint32_t field_timestamp = extract_timestamp(packet_data_at(0));

    // Verify ALL packets have the same timestamp
    for (size_t i = 0; i < num_packets; ++i) {
        EXPECT_EQ(extract_timestamp(packet_data_at(i)), field_timestamp)
            << "All packets within field must have identical timestamp";
    }
}

/* Test: Timing calculations for interlaced (ticks divided by 2) */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, InterlacedMode_TicksPerFieldCorrect)
{
    init_settings(2, 1, 128);
    configure_scan_type(VideoScanType::Interlaced);

    // For 25 fps interlaced (50 fields/sec) at 90kHz:
    // ticks_per_frame = 90000 / 25 = 3600
    // ticks_per_field = 3600 / 2 = 1800
    double expected_ticks_per_field = 1800.0;
    EXPECT_DOUBLE_EQ(rational_cast<double>(m_settings->ticks_per_media_unit),
                     expected_ticks_per_field)
        << "For 25fps interlaced (50 fields/sec), ticks_per_field should be 1800";

    // Verify media_unit_time_interval_ns is also halved
    double expected_field_time_ns = 1000000000.0 / 50.0;  // 20ms per field
    EXPECT_DOUBLE_EQ(m_settings->media_unit_time_interval_ns, expected_field_time_ns)
        << "For 50 fields/sec, field time should be 20ms";
}

/* Test: F-bit never has invalid value 0b01 */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, InterlacedMode_FBitNeverHasInvalidValue)
{
    init_settings(2, 1, 128);
    configure_scan_type(VideoScanType::Interlaced);
    init_writer();
    allocate_buffers(m_settings->packets_in_media_unit);

    // Test many fields to ensure we never see 0b01 (1) which is invalid
    for (int field = 0; field < 10; ++field) {
        set_media_unit(2, 10);

        size_t num_packets = m_writer->get_num_packets_for_next_chunk();
        EXPECT_EQ(write_chunk(num_packets), ReturnStatus::success);

        for (size_t i = 0; i < num_packets; ++i) {
            EXPECT_NE(extract_f_bit(packet_data_at(i)), 1)
                << "F-bit must never be 0b01 (invalid value)";
        }
    }
}
