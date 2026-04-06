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
#include <cmath>

#include <gtest/gtest.h>

#include "rdk/services/media/media_defs.h"
#include "rdk/services/media/media_settings_video.h"
#include "rdk/services/media/video_settings_calculator.h"
#include "rdk/services/settings/app_settings.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/utils/rational.h"

using namespace rdk::services;
using namespace testing;
using rdk::services::rational_cast;

/**
 * @brief: SMPTE 2110-20 video settings calculator test suite for interlaced video support.
 *
 * Tests cover field-based timing calculations, packet counts, and byte calculations.
 */
class VideoSettingsCalculatorTest : public ::testing::Test
{
protected:
    /** Create raw media settings without running the calculator. */
    void create_settings(VideoScanType scan_type, FrameRate frame_rate,
                         Resolution resolution = {_1080_WIDTH, _1080_HEIGHT})
    {
        m_settings.video_scan_type = scan_type;
        m_settings.frame_rate = frame_rate;
        m_settings.resolution = resolution;
        m_settings.sampling_type = VideoSampling::YCbCr_4_2_2;
        m_settings.bit_depth = VideoBitDepth::_10;
        m_settings.sample_rate = 90000;
    }

    /** Create media settings and run calculator, asserting success. */
    void init_settings(VideoScanType scan_type, FrameRate frame_rate,
                       Resolution resolution = {_1080_WIDTH, _1080_HEIGHT})
    {
        create_settings(scan_type, frame_rate, resolution);
        calculate_and_assert_success();
    }

    /** Run the calculator on m_settings and assert success. */
    void calculate_and_assert_success()
    {
        ST_2110_20_MediaSettingsCalculator calculator(m_settings);
        ASSERT_EQ(calculator.calculate_media_settings(), ReturnStatus::success);
    }

    /** Run the calculator on m_settings and return the status. */
    ReturnStatus calculate()
    {
        ST_2110_20_MediaSettingsCalculator calculator(m_settings);
        return calculator.calculate_media_settings();
    }

    SMPTE_2110_20_MediaSettings m_settings;
};

/* Test interlaced 1080i50 field-based calculations */
TEST_F(VideoSettingsCalculatorTest, Interlaced_1080i50_FieldBasedValues)
{
    init_settings(VideoScanType::Interlaced, FrameRate(25));

    // For 1080i50 (25 fps, 50 fields/sec):
    // - Full frame: 1920x1080, 25 fps
    // - Field: 1920x540, 50 fields/sec

    EXPECT_EQ(m_settings.lines_in_frame_field, 540)
        << "Lines in field should be 540 for 1080i";

    // 1 second / 50 fields = 20,000,000 ns per field
    EXPECT_NEAR(m_settings.media_unit_time_interval_ns, 20000000.0, 1.0)
        << "Field time should be 20ms for 1080i50";

    // RTP timestamp @ 90kHz: 90000 / 50 = 1800 ticks per field
    EXPECT_NEAR(rational_cast<double>(m_settings.ticks_per_media_unit), 1800.0, 1.0)
        << "RTP timestamp increment should be 1800 per field for 1080i50";

    // Packets per field = packets_in_line * 540
    size_t expected_packets_per_field = m_settings.packets_in_line * 540;
    EXPECT_EQ(m_settings.packets_in_media_unit, expected_packets_per_field)
        << "Packets per field should be half of full frame";

    // Bytes per field = 1920 * 540 * 2.5 (4:2:2 10-bit)
    size_t expected_bytes_per_field = 1920 * 540 * 5 / 2;
    EXPECT_EQ(m_settings.bytes_per_media_unit, expected_bytes_per_field)
        << "Bytes per field should be half of full frame";
}

/* Test interlaced 1080i60 fractional frame rate */
TEST_F(VideoSettingsCalculatorTest, Interlaced_1080i60_FractionalFrameRate)
{
    init_settings(VideoScanType::Interlaced, FrameRate(30000, 1001));

    // Field time for 1080i59.94 (29.97 fps frame = 59.94 fields/sec)
    double expected_field_time_ns = (NS_IN_SEC * 1001.0 / 30000.0) / 2.0;
    EXPECT_NEAR(m_settings.media_unit_time_interval_ns, expected_field_time_ns, 1.0)
        << "Field time should be ~16.68ms for 1080i59.94";

    // RTP timestamp increment: 90000 / 59.94 = 1501.5 ticks per field
    double expected_ticks = 90000.0 / (2.0 * 30000.0 / 1001.0);
    EXPECT_NEAR(rational_cast<double>(m_settings.ticks_per_media_unit), expected_ticks, 1.0)
        << "RTP timestamp increment should be ~1501.5 per field for 1080i59.94";

    EXPECT_EQ(m_settings.lines_in_frame_field, 540)
        << "Lines in field should be 540 for 1080i59.94";
}

/* Test progressive video remains unchanged */
TEST_F(VideoSettingsCalculatorTest, Progressive_1080p60_FrameBasedValues)
{
    init_settings(VideoScanType::Progressive, FrameRate(60));

    EXPECT_EQ(m_settings.lines_in_frame_field, 1080)
        << "Lines should be 1080 for progressive";

    // Frame time for 1080p60: 1 second / 60 = 16,666,667 ns
    EXPECT_NEAR(m_settings.media_unit_time_interval_ns, 16666666.67, 1.0)
        << "Frame time should be ~16.67ms for 1080p60";

    // RTP timestamp increment: 90000 / 60 = 1500 ticks per frame
    EXPECT_NEAR(rational_cast<double>(m_settings.ticks_per_media_unit), 1500.0, 1.0)
        << "RTP timestamp increment should be 1500 per frame for 1080p60";

    size_t expected_packets_per_frame = m_settings.packets_in_line * 1080;
    EXPECT_EQ(m_settings.packets_in_media_unit, expected_packets_per_frame)
        << "Packets should be for full frame in progressive";
}

/* Test get_packets_per_frame() multiplies by two for interlaced */
TEST_F(VideoSettingsCalculatorTest, Interlaced_GetPacketsPerFrame_MultipliesByTwo)
{
    init_settings(VideoScanType::Interlaced, FrameRate(25));

    size_t packets_per_field = m_settings.packets_in_media_unit;
    size_t packets_per_frame = m_settings.get_packets_per_frame();

    EXPECT_EQ(packets_per_frame, packets_per_field * 2)
        << "get_packets_per_frame() should return packets_in_media_unit * 2 for interlaced";

    EXPECT_EQ(packets_per_field, 2160)
        << "Packets per field should be 2160 for 1080i 4:2:2 10-bit";

    EXPECT_EQ(packets_per_frame, 4320)
        << "Packets per frame should be 4320 for 1080i 4:2:2 10-bit";
}

/* Test get_packets_per_frame() unchanged for progressive */
TEST_F(VideoSettingsCalculatorTest, Progressive_GetPacketsPerFrame_Unchanged)
{
    init_settings(VideoScanType::Progressive, FrameRate(60));

    EXPECT_EQ(m_settings.get_packets_per_frame(), m_settings.packets_in_media_unit)
        << "get_packets_per_frame() should equal packets_in_media_unit for progressive";

    EXPECT_EQ(m_settings.get_packets_per_frame(), 4320)
        << "Packets per frame should be 4320 for 1080p 4:2:2 10-bit";
}

/* Test progressive vs interlaced with same resolution and frame rate */
TEST_F(VideoSettingsCalculatorTest, ProgressiveVsInterlaced_CompareFieldToFrame)
{
    // Create both progressive 50fps and interlaced 25fps (50 fields/sec)
    SMPTE_2110_20_MediaSettings prog_settings;
    SMPTE_2110_20_MediaSettings intl_settings;

    // Progressive 50fps
    create_settings(VideoScanType::Progressive, FrameRate(50));
    prog_settings = m_settings;
    {
        ST_2110_20_MediaSettingsCalculator calc(prog_settings);
        ASSERT_EQ(calc.calculate_media_settings(), ReturnStatus::success);
    }

    // Interlaced 25fps (50 fields/sec)
    create_settings(VideoScanType::Interlaced, FrameRate(25));
    intl_settings = m_settings;
    {
        ST_2110_20_MediaSettingsCalculator calc(intl_settings);
        ASSERT_EQ(calc.calculate_media_settings(), ReturnStatus::success);
    }

    // Both should have same media unit time (20ms) since both are 50 media units/sec
    EXPECT_NEAR(prog_settings.media_unit_time_interval_ns,
                intl_settings.media_unit_time_interval_ns, 1.0)
        << "Progressive 50fps frame time should equal interlaced 25fps field time";

    EXPECT_NEAR(rational_cast<double>(prog_settings.ticks_per_media_unit),
                rational_cast<double>(intl_settings.ticks_per_media_unit), 1.0)
        << "Progressive 50fps and interlaced 25fps should have same RTP timestamp increment";

    EXPECT_EQ(intl_settings.lines_in_frame_field, prog_settings.lines_in_frame_field / 2)
        << "Interlaced field lines should be half of progressive frame lines";

    EXPECT_EQ(intl_settings.packets_in_media_unit, prog_settings.packets_in_media_unit / 2)
        << "Interlaced field packets should be half of progressive frame packets";
}

/* Test rejection of 4:2:0 sampling with interlaced video */
TEST_F(VideoSettingsCalculatorTest, Interlaced_Rejects_420_Sampling)
{
    create_settings(VideoScanType::Interlaced, FrameRate(25));
    m_settings.sampling_type = VideoSampling::YCbCr_4_2_0;

    EXPECT_EQ(calculate(), ReturnStatus::failure)
        << "4:2:0 sampling must be rejected with interlaced video";
}

/* Test interlaced accepts 4:4:4 sampling */
TEST_F(VideoSettingsCalculatorTest, Interlaced_Accepts_444_Sampling)
{
    create_settings(VideoScanType::Interlaced, FrameRate(25));
    m_settings.sampling_type = VideoSampling::YCbCr_4_4_4;
    calculate_and_assert_success();

    // For 1920x1080 4:4:4 10-bit: 3.75 bytes per pixel
    // Field: 1920 * 540 * 3.75 = 3,888,000 bytes
    size_t expected_bytes_per_field = 1920 * 540 * 15 / 4;
    EXPECT_EQ(m_settings.bytes_per_media_unit, expected_bytes_per_field)
        << "Bytes per field should be correct for 4:4:4 interlaced";
}

/* Test interlaced accepts RGB sampling */
TEST_F(VideoSettingsCalculatorTest, Interlaced_Accepts_RGB_Sampling)
{
    create_settings(VideoScanType::Interlaced, FrameRate(25));
    m_settings.sampling_type = VideoSampling::RGB;
    calculate_and_assert_success();

    // For RGB 10-bit: 3.75 bytes per pixel
    // Field: 1920 * 540 * 3.75 = 3,888,000 bytes
    size_t expected_bytes_per_field = 1920 * 540 * 15 / 4;
    EXPECT_EQ(m_settings.bytes_per_media_unit, expected_bytes_per_field)
        << "Bytes per field should be correct for RGB interlaced";
}

/* Test 1080i29.97 fractional frame rate */
TEST_F(VideoSettingsCalculatorTest, Interlaced_1080i29_97)
{
    init_settings(VideoScanType::Interlaced, FrameRate(30000, 1001));

    EXPECT_EQ(m_settings.lines_in_frame_field, 540)
        << "Lines in field should be 540 for 1080i29.97";

    double expected_field_time_ns = (NS_IN_SEC * 1001.0 / 30000.0) / 2.0;
    EXPECT_NEAR(m_settings.media_unit_time_interval_ns, expected_field_time_ns, 1.0)
        << "Field time should be ~16.68ms for 1080i29.97";
}

/* Test interlaced bytes per media unit for 8-bit depth */
TEST_F(VideoSettingsCalculatorTest, Interlaced_8bit_ByteCount)
{
    create_settings(VideoScanType::Interlaced, FrameRate(25));
    m_settings.bit_depth = VideoBitDepth::_8;
    calculate_and_assert_success();

    // For 1920x1080 4:2:2 8-bit: 2 bytes per pixel
    // Field: 1920 * 540 * 2 = 2,073,600 bytes
    EXPECT_EQ(m_settings.bytes_per_media_unit, static_cast<size_t>(1920 * 540 * 2))
        << "Bytes per field should be correct for 8-bit interlaced";
}

/* Test interlaced bytes per media unit for 12-bit depth */
TEST_F(VideoSettingsCalculatorTest, Interlaced_12bit_ByteCount)
{
    create_settings(VideoScanType::Interlaced, FrameRate(25));
    m_settings.bit_depth = VideoBitDepth::_12;
    calculate_and_assert_success();

    // For 1920x1080 4:2:2 12-bit: 3 bytes per pixel
    // Field: 1920 * 540 * 3 = 3,110,400 bytes
    EXPECT_EQ(m_settings.bytes_per_media_unit, static_cast<size_t>(1920 * 540 * 3))
        << "Bytes per field should be correct for 12-bit interlaced";
}
