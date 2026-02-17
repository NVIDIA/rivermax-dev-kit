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

#include <memory>

#include <gtest/gtest.h>

#include "rdk/services/ulp_packet_buffer/writers/rtp_smpte_2110_20_packet_buffer_writer.h"
#include "rdk/services/media/media_essence_source.h"
#include "rdk/services/media/media_defs.h"
#include "rdk/services/media/media_settings_video.h"
#include "rdk/services/media/video_settings_calculator.h"
#include "rdk/services/memory_allocation/new_memory_allocator.h"
#include "rdk/services/error_handling/return_status.h"

using namespace rivermax::dev_kit::services;
using namespace testing;

static std::shared_ptr<SMPTE_2110_20_MediaSettings> create_1080i50_settings()
{
    auto settings = std::make_shared<SMPTE_2110_20_MediaSettings>();
    settings->video_scan_type = VideoScanType::Interlaced;
    settings->frame_rate = FrameRate(25);  // 25 fps = 50 fields/sec
    settings->resolution = Resolution(_1080_WIDTH, _1080_HEIGHT);
    settings->sampling_type = VideoSampling::YCbCr_4_2_2;
    settings->bit_depth = VideoBitDepth::_10;
    settings->packet_payload_size = 1200;
    settings->sample_rate = 90000;  // 90 kHz RTP clock

    ST_2110_20_MediaSettingsCalculator calculator(*settings);
    if (calculator.calculate_media_settings() != ReturnStatus::success) {
        ADD_FAILURE() << "Failed to calculate 1080i50 media settings";
        return nullptr;
    }

    return settings;
}

static std::shared_ptr<MediaUnit> create_test_media_unit(size_t bytes_per_media_unit)
{
    auto media_unit = std::make_shared<MediaUnit>(bytes_per_media_unit, SMPTEStandard::ST_2110_20);
    // Fill with test pattern
    uint8_t* data = reinterpret_cast<uint8_t*>(media_unit->data->get());
    for (size_t i = 0; i < bytes_per_media_unit; ++i) {
        data[i] = static_cast<uint8_t>(i % 256);
    }
    return media_unit;
}

static uint8_t extract_field_indicator(const uint8_t* rtp_payload_header)
{
    // Skip Extended Sequence Number (2 bytes) and SRD Length (2 bytes)
    // F bit is the MSB of byte 4 (first byte of SRD Row Number field)
    return (rtp_payload_header[4] >> 7) & 0x01;
}

static uint16_t extract_line_number(const uint8_t* rtp_payload_header)
{
    // SRD Row Number is 15 bits starting from bit 1 of byte 4
    uint16_t line = ((rtp_payload_header[4] & 0x7F) << 8) | rtp_payload_header[5];
    return line;
}

static uint32_t extract_timestamp(const uint8_t* rtp_header)
{
    return (static_cast<uint32_t>(rtp_header[4]) << 24) |
           (static_cast<uint32_t>(rtp_header[5]) << 16) |
           (static_cast<uint32_t>(rtp_header[6]) << 8) |
           static_cast<uint32_t>(rtp_header[7]);
}

static uint8_t extract_marker_bit(const uint8_t* rtp_header)
{
    return (rtp_header[1] >> 7) & 0x01;
}

/**
 * @brief: Test suite for SMPTE 2110-20 interlaced video RTP packet generation.
 *
 * Tests verify compliance with SMPTE 2110-20 Section 6.1.4 requirements:
 * - Field Identification (F bit) alternation per field
 * - RTP timestamp increment per field
 * - Marker bit on last packet of field
 * - Line numbering starting at 0 per field
 */
class RTP_SMPTE_2110_20_PacketBufferWriterTest : public ::testing::Test
{
protected:
    static constexpr size_t MAX_PACKETS_PER_CHUNK = 100;

    void SetUp() override
    {
        m_mem_allocator = std::make_shared<NewMemoryAllocator>();
        m_header_mem_utils = m_mem_allocator->get_memory_utils();
        m_payload_mem_utils = m_mem_allocator->get_memory_utils();
    }

    /** Create 1080i50 settings and store in m_settings. */
    void init_settings()
    {
        m_settings = create_1080i50_settings();
        ASSERT_NE(m_settings, nullptr);
    }

    /** Create the packet buffer writer and store in m_writer. */
    void init_writer(bool enable_zero_copy = false)
    {
        m_writer = std::make_unique<RTP_SMPTE_2110_20_PacketBufferWriter>(
            *m_settings, m_header_mem_utils, m_payload_mem_utils, enable_zero_copy);
    }

    /** Allocate scratch buffer and payload sizes for the given number of packets. */
    void allocate_buffers(size_t num_packets)
    {
        m_scratch_buffer.resize(m_settings->data_stride_size * num_packets);
        m_payload_sizes.resize(num_packets);
    }

    /** Create a test media unit and set it on the writer. */
    void set_media_unit()
    {
        auto media_unit = create_test_media_unit(m_settings->bytes_per_media_unit);
        m_writer->set_next_media_unit(std::move(media_unit));
    }

    /** Write a chunk of packets into m_scratch_buffer/m_payload_sizes. */
    ReturnStatus write_chunk(size_t num_packets)
    {
        return m_writer->write_buffer(m_scratch_buffer.data(), num_packets,
                                      m_payload_sizes.data());
    }

    /** Return pointer to raw packet data at the given index in m_scratch_buffer. */
    uint8_t* packet_data_at(size_t packet_idx)
    {
        return m_scratch_buffer.data() + (packet_idx * m_settings->data_stride_size);
    }

    /**
     * Process all chunks for a complete field, calling per_packet_fn for each packet.
     * per_packet_fn receives (packet_data, global_packet_index).
     */
    template <typename Func>
    void process_all_field_chunks(Func per_packet_fn)
    {
        size_t packets_written = 0;
        while (packets_written < m_settings->packets_in_media_unit) {
            size_t num_packets = m_writer->get_num_packets_for_next_chunk();
            if (num_packets == 0) {
                break;
            }

            EXPECT_EQ(write_chunk(num_packets), ReturnStatus::success);

            for (size_t pkt = 0; pkt < num_packets; ++pkt) {
                per_packet_fn(packet_data_at(pkt), packets_written + pkt);
            }
            packets_written += num_packets;
        }
    }

    /** Consume all remaining chunks for the current field without inspecting packets. */
    void consume_remaining_field_chunks()
    {
        process_all_field_chunks([](uint8_t*, size_t) {});
    }

    std::shared_ptr<NewMemoryAllocator> m_mem_allocator;
    std::shared_ptr<MemoryUtils> m_header_mem_utils;
    std::shared_ptr<MemoryUtils> m_payload_mem_utils;
    std::shared_ptr<SMPTE_2110_20_MediaSettings> m_settings;
    std::unique_ptr<RTP_SMPTE_2110_20_PacketBufferWriter> m_writer;
    std::vector<uint8_t> m_scratch_buffer;
    std::vector<uint16_t> m_payload_sizes;
};

/* Test F-bit alternates between fields */
TEST_F(RTP_SMPTE_2110_20_PacketBufferWriterTest, Interlaced_FBitAlternatesBetweenFields)
{
    init_settings();
    init_writer();
    allocate_buffers(MAX_PACKETS_PER_CHUNK);

    // First field - all packets should have F=0
    set_media_unit();
    process_all_field_chunks([](uint8_t* packet, size_t) {
        EXPECT_EQ(extract_field_indicator(packet + RTP_HEADER_SIZE), 0)
            << "First field packet should have F=0";
    });

    // Second field - all packets should have F=1
    set_media_unit();
    process_all_field_chunks([](uint8_t* packet, size_t) {
        EXPECT_EQ(extract_field_indicator(packet + RTP_HEADER_SIZE), 1)
            << "Second field packet should have F=1";
    });

    // Third field should go back to F=0
    set_media_unit();
    process_all_field_chunks([](uint8_t* packet, size_t) {
        EXPECT_EQ(extract_field_indicator(packet + RTP_HEADER_SIZE), 0)
            << "Third field packet should have F=0 (cycle repeats)";
    });
}

/* Test RTP timestamp increments per field */
TEST_F(RTP_SMPTE_2110_20_PacketBufferWriterTest, Interlaced_TimestampIncrementsPerField)
{
    init_settings();
    init_writer();
    allocate_buffers(MAX_PACKETS_PER_CHUNK);

    uint32_t field1_timestamp = 0;
    uint32_t field2_timestamp = 0;
    uint32_t field3_timestamp = 0;

    // Field 1 - all packets should have same timestamp
    set_media_unit();
    process_all_field_chunks([&](uint8_t* packet, size_t global_idx) {
        uint32_t ts = extract_timestamp(packet);
        if (global_idx == 0) {
            field1_timestamp = ts;
        }
        EXPECT_EQ(ts, field1_timestamp) << "All packets in field 1 should have same timestamp";
    });

    // Field 2 - should increment by ticks_per_media_unit (1800 for 1080i50)
    set_media_unit();
    process_all_field_chunks([&](uint8_t* packet, size_t global_idx) {
        uint32_t ts = extract_timestamp(packet);
        if (global_idx == 0) {
            field2_timestamp = ts;
        }
        EXPECT_EQ(ts, field2_timestamp) << "All packets in field 2 should have same timestamp";
    });

    // Verify timestamp increment matches ticks_per_media_unit
    EXPECT_EQ(field2_timestamp - field1_timestamp,
              static_cast<uint32_t>(m_settings->ticks_per_media_unit))
        << "Timestamp should increment by ticks_per_media_unit ("
        << m_settings->ticks_per_media_unit << ") between fields";

    // Field 3 - verify increment again
    set_media_unit();
    process_all_field_chunks([&](uint8_t* packet, size_t global_idx) {
        uint32_t ts = extract_timestamp(packet);
        if (global_idx == 0) {
            field3_timestamp = ts;
        }
    });

    EXPECT_EQ(field3_timestamp - field2_timestamp,
              static_cast<uint32_t>(m_settings->ticks_per_media_unit))
        << "Timestamp should increment consistently between fields";
}

/* Test marker bit set on last packet of field */
TEST_F(RTP_SMPTE_2110_20_PacketBufferWriterTest, Interlaced_MarkerBitOnLastPacket)
{
    init_settings();
    init_writer();
    allocate_buffers(MAX_PACKETS_PER_CHUNK);
    set_media_unit();

    const size_t total_packets = m_settings->packets_in_media_unit;
    process_all_field_chunks([&](uint8_t* packet, size_t global_idx) {
        uint8_t marker = extract_marker_bit(packet);

        if (global_idx == total_packets - 1) {
            EXPECT_EQ(marker, 1) << "Last packet of field should have marker bit set to 1";
        } else {
            EXPECT_EQ(marker, 0) << "Non-last packet should have marker bit set to 0";
        }
    });
}

/* Test line numbering starts at 0 per field */
TEST_F(RTP_SMPTE_2110_20_PacketBufferWriterTest, Interlaced_LineNumberingStartsAtZero)
{
    init_settings();
    init_writer();
    allocate_buffers(m_settings->packets_in_media_unit);

    // First field - should start at line 0
    set_media_unit();

    size_t num_packets = m_writer->get_num_packets_for_next_chunk();
    EXPECT_EQ(write_chunk(num_packets), ReturnStatus::success);

    uint16_t first_line = extract_line_number(packet_data_at(0) + RTP_HEADER_SIZE);
    EXPECT_EQ(first_line, 0) << "First field should start at line 0";

    // Consume remaining packets of first field
    allocate_buffers(MAX_PACKETS_PER_CHUNK);
    consume_remaining_field_chunks();

    // Second field - should also start at line 0
    set_media_unit();

    num_packets = m_writer->get_num_packets_for_next_chunk();
    EXPECT_EQ(write_chunk(num_packets), ReturnStatus::success);

    first_line = extract_line_number(packet_data_at(0) + RTP_HEADER_SIZE);
    EXPECT_EQ(first_line, 0) << "Second field should also start at line 0";
}

/* Test field contains correct number of lines (540 for 1080i) */
TEST_F(RTP_SMPTE_2110_20_PacketBufferWriterTest, Interlaced_1080i_FieldHas540Lines)
{
    init_settings();

    // For 1080i, each field should have 540 lines (1080 / 2)
    EXPECT_EQ(m_settings->lines_in_frame_field, 540) << "1080i field should have 540 lines";

    init_writer();
    allocate_buffers(MAX_PACKETS_PER_CHUNK);
    set_media_unit();

    uint16_t max_line_number = 0;
    process_all_field_chunks([&](uint8_t* packet, size_t) {
        uint16_t line = extract_line_number(packet + RTP_HEADER_SIZE);
        if (line > max_line_number) {
            max_line_number = line;
        }
    });

    // Max line number should be 539 (lines 0-539 = 540 lines)
    EXPECT_EQ(max_line_number, 539) << "Max line number should be 539 for 540-line field";
}

/* Test SRD Row Number increments correctly within field */
TEST_F(RTP_SMPTE_2110_20_PacketBufferWriterTest, Interlaced_LineNumberIncrementsSequentially)
{
    init_settings();
    init_writer();
    allocate_buffers(MAX_PACKETS_PER_CHUNK);
    set_media_unit();

    uint16_t previous_line = 0;
    bool first_packet = true;

    process_all_field_chunks([&](uint8_t* packet, size_t) {
        uint16_t line = extract_line_number(packet + RTP_HEADER_SIZE);

        if (!first_packet) {
            // Line number should either stay the same (multiple packets per line)
            // or increment by 1 (moving to next line)
            EXPECT_TRUE(line == previous_line || line == previous_line + 1)
                << "Line number should increment by 0 or 1, got " << line
                << " after " << previous_line;
        }
        first_packet = false;
        previous_line = line;
    });
}

/* Test F-bit consistency across all packets in same field */
TEST_F(RTP_SMPTE_2110_20_PacketBufferWriterTest, Interlaced_FBitConsistentWithinField)
{
    init_settings();
    init_writer();
    allocate_buffers(MAX_PACKETS_PER_CHUNK);

    // Test multiple complete fields
    for (int field = 0; field < 4; ++field) {
        set_media_unit();

        uint8_t expected_f_bit = field % 2;  // Alternates 0, 1, 0, 1

        process_all_field_chunks([&](uint8_t* packet, size_t) {
            uint8_t f_bit = extract_field_indicator(packet + RTP_HEADER_SIZE);
            EXPECT_EQ(f_bit, expected_f_bit)
                << "Field " << field << " should have F=" << (int)expected_f_bit
                << " for all packets";
        });
    }
}

/* Test timestamp consistency across all packets in same field */
TEST_F(RTP_SMPTE_2110_20_PacketBufferWriterTest, Interlaced_TimestampConsistentWithinField)
{
    init_settings();
    init_writer();
    allocate_buffers(MAX_PACKETS_PER_CHUNK);
    set_media_unit();

    uint32_t field_timestamp = 0;
    bool first_packet = true;

    process_all_field_chunks([&](uint8_t* packet, size_t) {
        uint32_t ts = extract_timestamp(packet);

        if (first_packet) {
            field_timestamp = ts;
            first_packet = false;
        } else {
            EXPECT_EQ(ts, field_timestamp)
                << "All packets in same field must have identical timestamp";
        }
    });
}

/* Test marker bit is 0 for all non-last packets */
TEST_F(RTP_SMPTE_2110_20_PacketBufferWriterTest, Interlaced_MarkerBitZeroExceptLastPacket)
{
    init_settings();
    init_writer();
    allocate_buffers(MAX_PACKETS_PER_CHUNK);
    set_media_unit();

    const size_t total_packets = m_settings->packets_in_media_unit;
    size_t packets_with_marker = 0;

    process_all_field_chunks([&](uint8_t* packet, size_t global_idx) {
        uint8_t marker = extract_marker_bit(packet);

        if (marker == 1) {
            packets_with_marker++;
            // Marker bit should only be set on very last packet
            EXPECT_EQ(global_idx, total_packets - 1)
                << "Marker bit should only be set on last packet of field";
        }
    });

    // Exactly one packet should have marker bit set
    EXPECT_EQ(packets_with_marker, 1)
        << "Exactly one packet per field should have marker bit set";
}

/* Test multiple consecutive fields with correct sequence */
TEST_F(RTP_SMPTE_2110_20_PacketBufferWriterTest, Interlaced_MultipleFieldsCorrectSequence)
{
    init_settings();
    init_writer();
    allocate_buffers(MAX_PACKETS_PER_CHUNK);

    constexpr int NUM_FIELDS_TO_TEST = 10;
    uint32_t previous_timestamp = 0;
    bool first_field = true;

    for (int field = 0; field < NUM_FIELDS_TO_TEST; ++field) {
        set_media_unit();

        uint8_t expected_f_bit = field % 2;

        // Get first packet of this field
        size_t num_packets = m_writer->get_num_packets_for_next_chunk();
        ASSERT_GT(num_packets, 0);

        EXPECT_EQ(write_chunk(num_packets), ReturnStatus::success);

        uint8_t* first_packet = packet_data_at(0);
        uint8_t f_bit = extract_field_indicator(first_packet + RTP_HEADER_SIZE);
        uint32_t timestamp = extract_timestamp(first_packet);
        uint16_t line = extract_line_number(first_packet + RTP_HEADER_SIZE);

        // Verify F-bit alternation
        EXPECT_EQ(f_bit, expected_f_bit) << "Field " << field << " F-bit incorrect";

        // Verify line starts at 0
        EXPECT_EQ(line, 0) << "Field " << field << " should start at line 0";

        // Verify timestamp increments
        if (!first_field) {
            uint32_t increment = timestamp - previous_timestamp;
            EXPECT_EQ(increment, static_cast<uint32_t>(m_settings->ticks_per_media_unit))
                << "Timestamp increment incorrect between field " << (field - 1)
                << " and field " << field;
        }

        previous_timestamp = timestamp;
        first_field = false;

        // Consume remaining packets of this field
        consume_remaining_field_chunks();
    }
}

/* Test last line of field is correct */
TEST_F(RTP_SMPTE_2110_20_PacketBufferWriterTest, Interlaced_LastLineNumberIsCorrect)
{
    init_settings();

    // For 1080i, last line should be 539 (lines 0-539 = 540 lines)
    const uint16_t EXPECTED_LAST_LINE = m_settings->lines_in_frame_field - 1;

    init_writer();
    allocate_buffers(MAX_PACKETS_PER_CHUNK);
    set_media_unit();

    uint16_t last_line = 0;
    process_all_field_chunks([&](uint8_t* packet, size_t) {
        uint16_t line = extract_line_number(packet + RTP_HEADER_SIZE);
        last_line = line;
    });

    EXPECT_EQ(last_line, EXPECTED_LAST_LINE)
        << "Last packet should be from line " << EXPECTED_LAST_LINE
        << " (0-indexed line count = " << m_settings->lines_in_frame_field << ")";
}
