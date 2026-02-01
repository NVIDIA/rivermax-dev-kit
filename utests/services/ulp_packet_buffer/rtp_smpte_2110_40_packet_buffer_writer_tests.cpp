/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "rdk/services/media/media_essence_provider.h"
#include "rdk/services/media/media_defs.h"
#include "rdk/services/media/ancillary_metadata.h"
#include "rdk/services/media/ancillary_settings_calculator.h"
#include "rdk/services/memory_allocation/new_memory_allocator.h"

using namespace rivermax::dev_kit::services;
using namespace testing;

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

/**
 * @brief: SMPTE 2110-40 buffer writer test suite.
 *
 * Tests cover various scenarios of ancillary data packing into RTP packets.
 * Actual user data payload is not tested.
 */
class RTP_SMPTE_2110_40_PacketBufferWriterTest : public ::testing::Test
{
};

/* Test: Mock mode returns fixed packets_in_chunk for all chunks */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, MockMode_FixedPacketsPerChunk)
{
    constexpr uint32_t packets_in_media_unit = 6;
    constexpr size_t chunks_in_media_unit = 3;
    constexpr uint16_t max_user_data_words = 255;
    constexpr uint32_t max_ancillary_data_packets_per_packet = 10;
    constexpr bool enable_hds = false;
    auto settings =
        create_basic_settings(packets_in_media_unit, chunks_in_media_unit, max_user_data_words,
                              max_ancillary_data_packets_per_packet, enable_hds);
    ASSERT_NE(settings, nullptr);
    auto mem_allocator = std::make_shared<NewMemoryAllocator>();
    auto header_mem_utils = mem_allocator->get_memory_utils();
    auto payload_mem_utils = mem_allocator->get_memory_utils();

    constexpr bool enable_mock_mode = true;
    RTP_SMPTE_2110_40_PacketBufferWriter writer(*settings, header_mem_utils, payload_mem_utils,
                                                enable_mock_mode);

    // In mock mode, each chunk should return the same fixed number of packets
    constexpr size_t expected_packets_per_chunk = packets_in_media_unit / chunks_in_media_unit;
    std::vector<uint8_t> scratch_buffer(settings->data_stride_size * expected_packets_per_chunk);
    std::vector<uint16_t> payload_sizes(expected_packets_per_chunk);

    for (size_t chunk_idx = 0; chunk_idx < chunks_in_media_unit; ++chunk_idx) {
        EXPECT_EQ(writer.get_num_packets_for_next_chunk(), expected_packets_per_chunk)
            << "Chunk " << chunk_idx << " should return fixed packets_per_chunk in mock mode";

        ReturnStatus status = writer.write_buffer(scratch_buffer.data(), expected_packets_per_chunk,
                                                  payload_sizes.data());
        EXPECT_EQ(status, ReturnStatus::success) << "Chunk " << chunk_idx << " write_buffer failed";

        // Verify marker bit is only set on the last packet of the media unit
        for (size_t packet_idx = 0; packet_idx < expected_packets_per_chunk; ++packet_idx) {
            auto* rtp_header = reinterpret_cast<TestRTPHeader*>(
                scratch_buffer.data() + packet_idx * settings->data_stride_size);

            bool is_last_packet_of_media_unit = (chunk_idx == chunks_in_media_unit - 1) &&
                (packet_idx == expected_packets_per_chunk - 1);

            if (is_last_packet_of_media_unit) {
                EXPECT_EQ(rtp_header->marker, 1)
                    << "Last packet (chunk " << chunk_idx << ", packet " << packet_idx
                    << ") should have marker bit set";
            } else {
                EXPECT_EQ(rtp_header->marker, 0)
                    << "Non-last packet (chunk " << chunk_idx << ", packet " << packet_idx
                    << ") should NOT have marker bit set";
            }
        }
    }
}

/* Test: 3 small ancillary descriptors packed into 1 RTP packet, 1 chunk */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, ThreeAncillaryDescriptors_OneChunk)
{
    constexpr uint32_t packets_in_media_unit = 10;
    constexpr size_t chunks_in_media_unit = 1;
    constexpr uint16_t max_user_data_words = 255;
    constexpr uint32_t max_ancillary_data_packets_per_packet = 10;
    constexpr bool enable_hds = false;
    auto settings =
        create_basic_settings(packets_in_media_unit, chunks_in_media_unit, max_user_data_words,
                              max_ancillary_data_packets_per_packet, enable_hds);
    ASSERT_NE(settings, nullptr);

    constexpr size_t ancillary_data_packets_count = 3;
    constexpr uint16_t words_per_ancillary_data_packet = 10;
    auto media_unit = create_ancillary_media_unit(settings, ancillary_data_packets_count,
                                                  words_per_ancillary_data_packet);
    ASSERT_NE(media_unit, nullptr);
    auto mem_allocator = std::make_shared<NewMemoryAllocator>();
    auto header_mem_utils = mem_allocator->get_memory_utils();
    auto payload_mem_utils = mem_allocator->get_memory_utils();

    constexpr bool enable_mock_mode = false;
    RTP_SMPTE_2110_40_PacketBufferWriter writer(*settings, header_mem_utils, payload_mem_utils,
                                                enable_mock_mode);
    writer.set_next_media_unit(std::move(media_unit));

    //   Chunk 1: 1 packet with 3 ANC descriptors and Mark bit set
    constexpr size_t max_packets_per_chunk = packets_in_media_unit / chunks_in_media_unit;
    std::vector<uint8_t> scratch_buffer(settings->data_stride_size * max_packets_per_chunk);
    std::vector<uint16_t> payload_sizes(max_packets_per_chunk);

    constexpr size_t expected_packets_first_chunk = 1;
    EXPECT_EQ(writer.get_num_packets_for_next_chunk(), expected_packets_first_chunk);

    ReturnStatus status = writer.write_buffer(scratch_buffer.data(), expected_packets_first_chunk,
                                              payload_sizes.data());
    EXPECT_EQ(status, ReturnStatus::success);
    EXPECT_GT(payload_sizes[0], 0);

    auto* rtp_header = reinterpret_cast<TestRTPHeader*>(scratch_buffer.data());
    auto* ancillary_header =
        reinterpret_cast<TestAncillaryRTPExtension*>(scratch_buffer.data() + sizeof(TestRTPHeader));
    EXPECT_EQ(ancillary_header->anc_count, 3);
    EXPECT_EQ(rtp_header->marker, 1);
}

/* Test: 3 small ancillary descriptors packed into 1 RTP packet, 2 chunks */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, ThreeAncillaryDescriptors_TwoChunks)
{
    constexpr uint32_t packets_in_media_unit = 10;
    constexpr size_t chunks_in_media_unit = 2;
    constexpr uint16_t max_user_data_words = 255;
    constexpr uint32_t max_ancillary_data_packets_per_packet = 10;
    constexpr bool enable_hds = false;
    auto settings =
        create_basic_settings(packets_in_media_unit, chunks_in_media_unit, max_user_data_words,
                              max_ancillary_data_packets_per_packet, enable_hds);
    ASSERT_NE(settings, nullptr);

    constexpr size_t ancillary_data_packets_count = 3;
    constexpr uint16_t words_per_ancillary_data_packet = 10;
    auto media_unit = create_ancillary_media_unit(settings, ancillary_data_packets_count,
                                                  words_per_ancillary_data_packet);
    ASSERT_NE(media_unit, nullptr);
    auto mem_allocator = std::make_shared<NewMemoryAllocator>();
    auto header_mem_utils = mem_allocator->get_memory_utils();
    auto payload_mem_utils = mem_allocator->get_memory_utils();

    constexpr bool enable_mock_mode = false;
    RTP_SMPTE_2110_40_PacketBufferWriter writer(*settings, header_mem_utils, payload_mem_utils,
                                                enable_mock_mode);
    writer.set_next_media_unit(std::move(media_unit));

    //   Chunk 1: 1 packet with 3 ancillary descriptors
    //   Chunk 2: 1 packet with 0 ancillary descriptors and marker bit set
    constexpr size_t max_packets_per_chunk = packets_in_media_unit / chunks_in_media_unit;
    std::vector<uint8_t> scratch_buffer(settings->data_stride_size * max_packets_per_chunk);
    std::vector<uint16_t> payload_sizes(max_packets_per_chunk);

    constexpr size_t expected_packets_first_chunk = 1;
    EXPECT_EQ(writer.get_num_packets_for_next_chunk(), expected_packets_first_chunk);

    ReturnStatus status = writer.write_buffer(scratch_buffer.data(), expected_packets_first_chunk,
                                              payload_sizes.data());
    EXPECT_EQ(status, ReturnStatus::success);
    EXPECT_GT(payload_sizes[0], 0);

    auto* rtp_header = reinterpret_cast<TestRTPHeader*>(scratch_buffer.data());
    auto* ancillary_header =
        reinterpret_cast<TestAncillaryRTPExtension*>(scratch_buffer.data() + sizeof(TestRTPHeader));
    EXPECT_EQ(ancillary_header->anc_count, 3);
    EXPECT_EQ(rtp_header->marker, 0);

    // Second chunk should have 1 packet with 0 data and Mark bit set
    size_t expected_packets_second_chunk = 1;
    EXPECT_GE(writer.get_num_packets_for_next_chunk(), expected_packets_second_chunk);

    status = writer.write_buffer(scratch_buffer.data(), expected_packets_second_chunk,
                                 payload_sizes.data());
    EXPECT_EQ(status, ReturnStatus::success);

    rtp_header = reinterpret_cast<TestRTPHeader*>(scratch_buffer.data());
    ancillary_header =
        reinterpret_cast<TestAncillaryRTPExtension*>(scratch_buffer.data() + sizeof(TestRTPHeader));
    EXPECT_EQ(ancillary_header->anc_count, 0);
    EXPECT_EQ(rtp_header->marker, 1);
}

/* Test: 3 small ancillary descriptors packed into 2 RTP packets, 1 chunk */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, ThreeAncillaryDescriptors_TwoRTPPackets_OneChunk)
{
    constexpr uint32_t packets_in_media_unit = 10;
    constexpr size_t chunks_in_media_unit = 1;
    constexpr uint16_t max_user_data_words = 255;
    constexpr uint32_t max_ancillary_data_packets_per_packet = 2;
    constexpr bool enable_hds = false;
    auto settings =
        create_basic_settings(packets_in_media_unit, chunks_in_media_unit, max_user_data_words,
                              max_ancillary_data_packets_per_packet, enable_hds);
    ASSERT_NE(settings, nullptr);

    constexpr size_t ancillary_data_packets_count = 3;
    constexpr uint16_t words_per_ancillary_data_packet = 10;
    auto media_unit = create_ancillary_media_unit(settings, ancillary_data_packets_count,
                                                  words_per_ancillary_data_packet);
    ASSERT_NE(media_unit, nullptr);
    auto mem_allocator = std::make_shared<NewMemoryAllocator>();
    auto header_mem_utils = mem_allocator->get_memory_utils();
    auto payload_mem_utils = mem_allocator->get_memory_utils();

    constexpr bool enable_mock_mode = false;
    RTP_SMPTE_2110_40_PacketBufferWriter writer(*settings, header_mem_utils, payload_mem_utils,
                                                enable_mock_mode);
    writer.set_next_media_unit(std::move(media_unit));

    // 3 ancillary descriptors, max 2 per RTP packet → packs into 2 RTP packets
    // With 1 chunk:
    //   Chunk 1: 2 RTP packets with 2+1 descriptors
    constexpr size_t max_packets_per_chunk = packets_in_media_unit / chunks_in_media_unit;
    std::vector<uint8_t> scratch_buffer(settings->data_stride_size * max_packets_per_chunk);
    std::vector<uint16_t> payload_sizes(max_packets_per_chunk);

    constexpr size_t expected_packets_first_chunk = 2;
    EXPECT_EQ(writer.get_num_packets_for_next_chunk(), expected_packets_first_chunk);

    ReturnStatus status = writer.write_buffer(scratch_buffer.data(), expected_packets_first_chunk,
                                              payload_sizes.data());
    EXPECT_EQ(status, ReturnStatus::success);
    EXPECT_GT(payload_sizes[0], 0);
    EXPECT_GT(payload_sizes[1], 0);

    // Chunk 1: First packet: should have 2 descriptors, Mark bit NOT set
    auto* rtp_header = reinterpret_cast<TestRTPHeader*>(scratch_buffer.data());
    auto* ancillary_header =
        reinterpret_cast<TestAncillaryRTPExtension*>(scratch_buffer.data() + sizeof(TestRTPHeader));
    EXPECT_EQ(ancillary_header->anc_count, 2);
    EXPECT_EQ(rtp_header->marker, 0);

    // Chunk 1: Second packet: should have 1 descriptor, Mark bit set
    rtp_header =
        reinterpret_cast<TestRTPHeader*>(scratch_buffer.data() + settings->data_stride_size);
    ancillary_header = reinterpret_cast<TestAncillaryRTPExtension*>(
        scratch_buffer.data() + settings->data_stride_size + sizeof(TestRTPHeader));
    EXPECT_EQ(ancillary_header->anc_count, 1);
    EXPECT_EQ(rtp_header->marker, 1);
}

/* Test: 3 small ancillary descriptors packed into 2 RTP packets, too many chunks (3) */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest,
       ThreeAncillaryDescriptors_TwoRTPPackets_TooManyChunks)
{
    constexpr uint32_t packets_in_media_unit = 10;
    constexpr size_t chunks_in_media_unit = 3;
    constexpr uint16_t max_user_data_words = 255;
    constexpr uint32_t max_ancillary_data_packets_per_packet = 2;
    constexpr bool enable_hds = false;
    auto settings =
        create_basic_settings(packets_in_media_unit, chunks_in_media_unit, max_user_data_words,
                              max_ancillary_data_packets_per_packet, enable_hds);
    ASSERT_NE(settings, nullptr);

    constexpr size_t ancillary_data_packets_count = 3;
    constexpr uint16_t words_per_ancillary_data_packet = 10;
    auto media_unit = create_ancillary_media_unit(settings, ancillary_data_packets_count,
                                                  words_per_ancillary_data_packet);
    ASSERT_NE(media_unit, nullptr);
    auto mem_allocator = std::make_shared<NewMemoryAllocator>();
    auto header_mem_utils = mem_allocator->get_memory_utils();
    auto payload_mem_utils = mem_allocator->get_memory_utils();

    constexpr bool enable_mock_mode = false;
    RTP_SMPTE_2110_40_PacketBufferWriter writer(*settings, header_mem_utils, payload_mem_utils,
                                                enable_mock_mode);
    writer.set_next_media_unit(std::move(media_unit));

    // 3 ancillary descriptors, max 2 per RTP packet → packs into 2 RTP packets
    // With 3 chunks:
    //   Chunk 1: 2 RTP packets with Mark bit not set
    //   Chunk 2: 1 empty RTP packet with Mark bit set
    //   Chunk 3: 1 empty RTP packet with Mark bit set
    constexpr size_t max_packets_per_chunk = packets_in_media_unit / chunks_in_media_unit;
    std::vector<uint8_t> scratch_buffer(settings->data_stride_size * max_packets_per_chunk);
    std::vector<uint16_t> payload_sizes(max_packets_per_chunk);

    constexpr size_t expected_packets_first_chunk = 2;
    EXPECT_EQ(writer.get_num_packets_for_next_chunk(), expected_packets_first_chunk);

    ReturnStatus status = writer.write_buffer(scratch_buffer.data(), expected_packets_first_chunk,
                                              payload_sizes.data());
    EXPECT_EQ(status, ReturnStatus::success);
    EXPECT_GT(payload_sizes[0], 0);
    EXPECT_GT(payload_sizes[1], 0);

    // Chunk 1: First packet: should have 2 descriptors, Mark bit NOT set
    auto* rtp_header = reinterpret_cast<TestRTPHeader*>(scratch_buffer.data());
    auto* ancillary_header =
        reinterpret_cast<TestAncillaryRTPExtension*>(scratch_buffer.data() + sizeof(TestRTPHeader));
    EXPECT_EQ(ancillary_header->anc_count, 2);
    EXPECT_EQ(rtp_header->marker, 0);

    // Chunk 1: Second packet: should have 1 descriptor, Mark bit NOT set
    rtp_header =
        reinterpret_cast<TestRTPHeader*>(scratch_buffer.data() + settings->data_stride_size);
    ancillary_header = reinterpret_cast<TestAncillaryRTPExtension*>(
        scratch_buffer.data() + settings->data_stride_size + sizeof(TestRTPHeader));
    EXPECT_EQ(ancillary_header->anc_count, 1);
    EXPECT_EQ(rtp_header->marker, 0);

    constexpr size_t expected_packets_second_chunk = 1;
    EXPECT_EQ(writer.get_num_packets_for_next_chunk(), expected_packets_second_chunk);

    status = writer.write_buffer(scratch_buffer.data(), expected_packets_second_chunk,
                                 payload_sizes.data());
    EXPECT_EQ(status, ReturnStatus::success);

    // Chunk 2: Third packet: should have 0 descriptors, Mark bit NOT set
    rtp_header = reinterpret_cast<TestRTPHeader*>(scratch_buffer.data());
    ancillary_header =
        reinterpret_cast<TestAncillaryRTPExtension*>(scratch_buffer.data() + sizeof(TestRTPHeader));
    EXPECT_EQ(ancillary_header->anc_count, 0);
    EXPECT_EQ(rtp_header->marker, 0);

    constexpr size_t expected_packets_third_chunk = 1;
    EXPECT_EQ(writer.get_num_packets_for_next_chunk(), expected_packets_third_chunk);

    status = writer.write_buffer(scratch_buffer.data(), expected_packets_third_chunk,
                                 payload_sizes.data());
    EXPECT_EQ(status, ReturnStatus::success);

    // Chunk 3: Fourth packet: should have 0 descriptors, Mark bit set
    rtp_header = reinterpret_cast<TestRTPHeader*>(scratch_buffer.data());
    ancillary_header =
        reinterpret_cast<TestAncillaryRTPExtension*>(scratch_buffer.data() + sizeof(TestRTPHeader));
    EXPECT_EQ(ancillary_header->anc_count, 0);
    EXPECT_EQ(rtp_header->marker, 1);
}

/* Test: 10 ancillary descriptors with 50 user data words - all fit in 1 RTP packet */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, TenANC_50Words_FitsInOneRTPPacket)
{
    constexpr uint32_t packets_in_media_unit = 10;
    constexpr size_t chunks_in_media_unit = 1;
    constexpr uint16_t max_user_data_words = 255;
    constexpr uint32_t max_ancillary_data_packets_per_packet = 10;
    constexpr bool enable_hds = false;
    auto settings =
        create_basic_settings(packets_in_media_unit, chunks_in_media_unit, max_user_data_words,
                              max_ancillary_data_packets_per_packet, enable_hds);
    ASSERT_NE(settings, nullptr);

    constexpr size_t ancillary_data_packets_count = 10;
    constexpr uint16_t words_per_ancillary_data_packet = 50;
    auto media_unit = create_ancillary_media_unit(settings, ancillary_data_packets_count,
                                                  words_per_ancillary_data_packet);
    ASSERT_NE(media_unit, nullptr);

    auto mem_allocator = std::make_shared<NewMemoryAllocator>();
    auto header_mem_utils = mem_allocator->get_memory_utils();
    auto payload_mem_utils = mem_allocator->get_memory_utils();

    constexpr bool enable_mock_mode = false;
    RTP_SMPTE_2110_40_PacketBufferWriter writer(*settings, header_mem_utils, payload_mem_utils,
                                                enable_mock_mode);
    writer.set_next_media_unit(std::move(media_unit));

    constexpr size_t expected_rtp_packets = 1;
    EXPECT_EQ(writer.get_num_packets_for_next_chunk(), expected_rtp_packets);

    std::vector<uint8_t> scratch_buffer(settings->data_stride_size * expected_rtp_packets);
    std::vector<uint16_t> payload_sizes(expected_rtp_packets);

    ReturnStatus status =
        writer.write_buffer(scratch_buffer.data(), expected_rtp_packets, payload_sizes.data());
    EXPECT_EQ(status, ReturnStatus::success);
    EXPECT_GT(payload_sizes[0], 0);

    // Verify the single packet has all 10 descriptors and marker bit set
    auto* rtp_header = reinterpret_cast<TestRTPHeader*>(scratch_buffer.data());
    auto* ancillary_header =
        reinterpret_cast<TestAncillaryRTPExtension*>(scratch_buffer.data() + sizeof(TestRTPHeader));
    EXPECT_EQ(ancillary_header->anc_count, 10) << "Single packet should contain all 10 ancillary descriptors";
    EXPECT_EQ(rtp_header->marker, 1) << "Single packet should have marker bit set";
}

/* Test: 10 ANC with 255 user data words - maximum size constraint */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, TenANC_255Words_MaxPayloadSizeConstrained)
{
    constexpr uint32_t packets_in_media_unit = 10;
    constexpr size_t chunks_in_media_unit = 1;
    constexpr uint16_t max_user_data_words = 255;
    constexpr uint32_t max_ancillary_data_packets_per_packet = 10;
    constexpr bool enable_hds = false;
    auto settings =
        create_basic_settings(packets_in_media_unit, chunks_in_media_unit, max_user_data_words,
                              max_ancillary_data_packets_per_packet, enable_hds);
    ASSERT_NE(settings, nullptr);

    constexpr size_t ancillary_data_packets_count = 10;
    constexpr uint16_t words_per_ancillary_data_packet = 255;
    auto media_unit = create_ancillary_media_unit(settings, ancillary_data_packets_count,
                                                  words_per_ancillary_data_packet);
    ASSERT_NE(media_unit, nullptr);

    auto mem_allocator = std::make_shared<NewMemoryAllocator>();
    auto header_mem_utils = mem_allocator->get_memory_utils();
    auto payload_mem_utils = mem_allocator->get_memory_utils();

    constexpr bool enable_mock_mode = false;
    RTP_SMPTE_2110_40_PacketBufferWriter writer(*settings, header_mem_utils, payload_mem_utils,
                                                enable_mock_mode);
    writer.set_next_media_unit(std::move(media_unit));

    // 10 ancillary descriptors with 255 user data words each
    // Expects 3 full RTP packets to pack all user data words
    constexpr size_t expected_rtp_packets = 3;
    EXPECT_EQ(writer.get_num_packets_for_next_chunk(), expected_rtp_packets);

    std::vector<uint8_t> scratch_buffer(settings->data_stride_size * expected_rtp_packets);
    std::vector<uint16_t> payload_sizes(expected_rtp_packets);

    ReturnStatus status =
        writer.write_buffer(scratch_buffer.data(), expected_rtp_packets, payload_sizes.data());
    EXPECT_EQ(status, ReturnStatus::success);

    // Verify all RTP packets were written
    for (size_t i = 0; i < expected_rtp_packets; ++i) {
        EXPECT_GT(payload_sizes[i], 0) << "Packet " << i << " should have non-zero payload";
    }

    // Verify last RTP packet has marker bit set
    auto* last_rtp_header = reinterpret_cast<TestRTPHeader*>(
        scratch_buffer.data() + settings->data_stride_size * (expected_rtp_packets - 1));
    EXPECT_EQ(last_rtp_header->marker, 1) << "Last packet should have marker bit set";

    // Verify that each RTP packet respects the packet payload size limit
    for (size_t i = 0; i < expected_rtp_packets; ++i) {
        EXPECT_LE(payload_sizes[i], settings->packet_payload_size)
            << "Packet " << i << " exceeds max payload size";
    }
}

/* Test: 10 ancillary descriptors with 50 user data words - all fit in 1 RTP packet - HDS mode */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, TenANC_50Words_FitsInOneRTPPacket_HDS)
{
    constexpr uint32_t packets_in_media_unit = 10;
    constexpr size_t chunks_in_media_unit = 1;
    constexpr uint16_t max_user_data_words = 255;
    constexpr uint32_t max_ancillary_data_packets_per_packet = 10;
    constexpr bool enable_hds = true;
    auto settings =
        create_basic_settings(packets_in_media_unit, chunks_in_media_unit, max_user_data_words,
                              max_ancillary_data_packets_per_packet, enable_hds);
    ASSERT_NE(settings, nullptr);

    constexpr size_t ancillary_data_packets_count = 10;
    constexpr uint16_t words_per_ancillary_data_packet = 50;
    auto media_unit = create_ancillary_media_unit(settings, ancillary_data_packets_count,
                                                  words_per_ancillary_data_packet);
    ASSERT_NE(media_unit, nullptr);

    auto mem_allocator = std::make_shared<NewMemoryAllocator>();
    auto header_mem_utils = mem_allocator->get_memory_utils();
    auto payload_mem_utils = mem_allocator->get_memory_utils();

    constexpr bool enable_mock_mode = false;
    RTP_SMPTE_2110_40_PacketBufferWriter writer(*settings, header_mem_utils, payload_mem_utils,
                                                enable_mock_mode);
    writer.set_next_media_unit(std::move(media_unit));

    constexpr size_t expected_rtp_packets = 1;
    EXPECT_EQ(writer.get_num_packets_for_next_chunk(), expected_rtp_packets);

    std::vector<uint8_t> header_buffer(settings->app_header_stride_size * expected_rtp_packets);
    std::vector<uint8_t> payload_buffer(settings->data_stride_size * expected_rtp_packets);
    std::vector<uint16_t> header_sizes(expected_rtp_packets, 0);
    std::vector<uint16_t> payload_sizes(expected_rtp_packets, 0);

    ReturnStatus status =
        writer.write_buffer(header_buffer.data(), payload_buffer.data(), expected_rtp_packets, header_sizes.data(), payload_sizes.data());
    EXPECT_EQ(status, ReturnStatus::success);
    EXPECT_GT(header_sizes[0], 0);
    EXPECT_GT(payload_sizes[0], 0);

    // Verify the single packet has all 10 descriptors and marker bit set
    auto* rtp_header = reinterpret_cast<TestRTPHeader*>(header_buffer.data());
    auto* ancillary_header =
        reinterpret_cast<TestAncillaryRTPExtension*>(header_buffer.data() + sizeof(TestRTPHeader));
    EXPECT_EQ(ancillary_header->anc_count, 10) << "Single packet should contain all 10 ancillary descriptors";
    EXPECT_EQ(rtp_header->marker, 1) << "Single packet should have marker bit set";
}

/* Test: 10 ANC with 255 user data words - maximum size constraint - HDS mode */
TEST_F(RTP_SMPTE_2110_40_PacketBufferWriterTest, TenANC_255Words_MaxPayloadSizeConstrained_HDS)
{
    constexpr uint32_t packets_in_media_unit = 10;
    constexpr size_t chunks_in_media_unit = 1;
    constexpr uint16_t max_user_data_words = 255;
    constexpr uint32_t max_ancillary_data_packets_per_packet = 10;
    constexpr bool enable_hds = true;
    auto settings =
        create_basic_settings(packets_in_media_unit, chunks_in_media_unit, max_user_data_words,
                              max_ancillary_data_packets_per_packet, enable_hds);
    ASSERT_NE(settings, nullptr);

    constexpr size_t ancillary_data_packets_count = 10;
    constexpr uint16_t words_per_ancillary_data_packet = 255;
    auto media_unit = create_ancillary_media_unit(settings, ancillary_data_packets_count,
                                                  words_per_ancillary_data_packet);
    ASSERT_NE(media_unit, nullptr);

    auto mem_allocator = std::make_shared<NewMemoryAllocator>();
    auto header_mem_utils = mem_allocator->get_memory_utils();
    auto payload_mem_utils = mem_allocator->get_memory_utils();

    constexpr bool enable_mock_mode = false;
    RTP_SMPTE_2110_40_PacketBufferWriter writer(*settings, header_mem_utils, payload_mem_utils,
                                                enable_mock_mode);
    writer.set_next_media_unit(std::move(media_unit));

    // 10 ancillary descriptors with 255 user data words each
    // Expects 3 full RTP packets to pack all user data words
    constexpr size_t expected_rtp_packets = 3;
    EXPECT_EQ(writer.get_num_packets_for_next_chunk(), expected_rtp_packets);

    std::vector<uint8_t> header_buffer(settings->app_header_stride_size * expected_rtp_packets);
    std::vector<uint8_t> payload_buffer(settings->data_stride_size * expected_rtp_packets);
    std::vector<uint16_t> header_sizes(expected_rtp_packets, 0);
    std::vector<uint16_t> payload_sizes(expected_rtp_packets, 0);

    ReturnStatus status =
        writer.write_buffer(header_buffer.data(), payload_buffer.data(), expected_rtp_packets, header_sizes.data(), payload_sizes.data());
    EXPECT_EQ(status, ReturnStatus::success);

    // Verify all RTP packets were written
    for (size_t i = 0; i < expected_rtp_packets; ++i) {
        EXPECT_GT(header_sizes[i], 0) << "Packet " << i << " should have non-zero header";
        EXPECT_GT(payload_sizes[i], 0) << "Packet " << i << " should have non-zero payload";
    }

    // Verify last RTP packet has marker bit set
    auto* last_rtp_header = reinterpret_cast<TestRTPHeader*>(
        header_buffer.data() + settings->app_header_stride_size * (expected_rtp_packets - 1));
    EXPECT_EQ(last_rtp_header->marker, 1) << "Last packet should have marker bit set";

    // Verify that each RTP packet respects the packet payload size limit
    for (size_t i = 0; i < expected_rtp_packets; ++i) {
        EXPECT_LE(payload_sizes[i], settings->packet_payload_size)
            << "Packet " << i << " exceeds max payload size";
    }
}
