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

#include <cmath>
#include <memory>
#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include "rdk/services/media/media_defs.h"
#include "rdk/services/media/media_settings_ancillary.h"
#include "rdk/services/media/ancillary_settings_calculator.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/utils/rational.h"

using namespace rivermax::dev_kit::services;
using namespace testing;
using rivermax::dev_kit::services::rational_cast;

/**
 * @brief: SMPTE 2110-40 ancillary settings calculator test suite for interlaced video support.
 *
 * Tests cover field-based timing calculations and F-bit compliance.
 */
class AncillarySettingsCalculatorTest : public ::testing::Test
{
protected:
    /** Create raw media settings without running the calculator. */
    void create_settings(VideoScanType scan_type, FrameRate frame_rate)
    {
        m_settings = SMPTE_2110_40_MediaSettings{};
        m_settings.video_scan_type = scan_type;
        m_settings.frame_rate = frame_rate;
        m_settings.data_identifiers = {{0x41, 0x05}};  // AFD
        m_settings.max_user_data_words_count = 255;
        m_settings.max_ancillary_data_packets_per_packet = 10;
    }

    /** Create media settings and run calculator, asserting success. */
    void init_settings(VideoScanType scan_type, FrameRate frame_rate)
    {
        create_settings(scan_type, frame_rate);
        calculate_and_assert_success();
    }

    /** Run the calculator on m_settings and assert success. */
    void calculate_and_assert_success()
    {
        ST_2110_40_MediaSettingsCalculator calculator(m_settings);
        ASSERT_EQ(calculator.calculate_media_settings(), ReturnStatus::success);
    }

    /** Run the calculator on m_settings and return the status. */
    ReturnStatus calculate()
    {
        ST_2110_40_MediaSettingsCalculator calculator(m_settings);
        return calculator.calculate_media_settings();
    }

    SMPTE_2110_40_MediaSettings m_settings;
};

/** Parameterized fixture for interlaced field timing (one param: fps). */
class AncillarySettingsCalculatorTest_FieldTimingDividedByTwo
    : public AncillarySettingsCalculatorTest,
      public testing::WithParamInterface<uint32_t> {};

/** Parameterized fixture for interlaced common frame rates (fps, expected_field_time_ms, expected_ticks). */
using CommonFrameRateParam = std::tuple<uint32_t, double, double>;
class AncillarySettingsCalculatorTest_CommonFrameRates
    : public AncillarySettingsCalculatorTest,
      public testing::WithParamInterface<CommonFrameRateParam> {};

/* Test interlaced 1080i50 field-based timing calculations */
TEST_F(AncillarySettingsCalculatorTest, Interlaced_1080i50_FieldBasedTiming)
{
    init_settings(VideoScanType::Interlaced, FrameRate(25));

    // For 1080i50 (25 fps, 50 fields/sec):
    // 1 second / 50 fields = 20,000,000 ns per field
    EXPECT_NEAR(m_settings.media_unit_time_interval_ns, 20000000.0, 1.0)
        << "Field time should be 20ms for 1080i50";

    // RTP timestamp @ 90kHz: 90000 / 50 = 1800 ticks per field
    EXPECT_NEAR(rational_cast<double>(m_settings.ticks_per_media_unit), 1800.0, 1.0)
        << "RTP timestamp increment should be 1800 per field for 1080i50";
}

/* Test interlaced 1080i60 fractional frame rate */
TEST_F(AncillarySettingsCalculatorTest, Interlaced_1080i60_FractionalFrameRate)
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
}

/* Test progressive ancillary timing remains unchanged */
TEST_F(AncillarySettingsCalculatorTest, Progressive_1080p60_FrameBasedTiming)
{
    init_settings(VideoScanType::Progressive, FrameRate(60));

    // Frame time for 1080p60: 1 second / 60 = 16,666,667 ns
    EXPECT_NEAR(m_settings.media_unit_time_interval_ns, 16666666.67, 1.0)
        << "Frame time should be ~16.67ms for 1080p60";

    // RTP timestamp increment: 90000 / 60 = 1500 ticks per frame
    EXPECT_NEAR(rational_cast<double>(m_settings.ticks_per_media_unit), 1500.0, 1.0)
        << "RTP timestamp increment should be 1500 per frame for 1080p60";
}

/* Test progressive vs interlaced with same field/frame rate */
TEST_F(AncillarySettingsCalculatorTest, ProgressiveVsInterlaced_CompareFieldToFrame)
{
    // Create both progressive 50fps and interlaced 25fps (50 fields/sec)
    SMPTE_2110_40_MediaSettings prog_settings;
    SMPTE_2110_40_MediaSettings intl_settings;

    create_settings(VideoScanType::Progressive, FrameRate(50));
    prog_settings = m_settings;
    {
        ST_2110_40_MediaSettingsCalculator calc(prog_settings);
        ASSERT_EQ(calc.calculate_media_settings(), ReturnStatus::success);
    }

    create_settings(VideoScanType::Interlaced, FrameRate(25));
    intl_settings = m_settings;
    {
        ST_2110_40_MediaSettingsCalculator calc(intl_settings);
        ASSERT_EQ(calc.calculate_media_settings(), ReturnStatus::success);
    }

    EXPECT_NEAR(prog_settings.media_unit_time_interval_ns,
                intl_settings.media_unit_time_interval_ns, 1.0)
        << "Progressive 50fps frame time should equal interlaced 25fps field time";

    EXPECT_NEAR(rational_cast<double>(prog_settings.ticks_per_media_unit),
                rational_cast<double>(intl_settings.ticks_per_media_unit), 1.0)
        << "Progressive 50fps and interlaced 25fps should have same RTP timestamp increment";
}

/* Test interlaced 1080i29.97 fractional frame rate */
TEST_F(AncillarySettingsCalculatorTest, Interlaced_1080i29_97)
{
    init_settings(VideoScanType::Interlaced, FrameRate(30000, 1001));

    double expected_field_time_ns = (NS_IN_SEC * 1001.0 / 30000.0) / 2.0;
    EXPECT_NEAR(m_settings.media_unit_time_interval_ns, expected_field_time_ns, 1.0)
        << "Field time should be ~16.68ms for 1080i29.97";
}

/* Test interlaced field timing divided by 2 from frame timing (parameterized by fps) */
TEST_P(AncillarySettingsCalculatorTest_FieldTimingDividedByTwo, FieldTimeAndTicksHalfOfFrame)
{
    const uint32_t fps = GetParam();
    init_settings(VideoScanType::Interlaced, FrameRate(fps));

    double expected_field_time_ns = (NS_IN_SEC / fps) / 2.0;
    EXPECT_NEAR(m_settings.media_unit_time_interval_ns, expected_field_time_ns, 1.0)
        << "Field time should be half of frame time for " << fps << " fps";

    double expected_field_ticks = (90000.0 / fps) / 2.0;
    EXPECT_NEAR(rational_cast<double>(m_settings.ticks_per_media_unit), expected_field_ticks, 1.0)
        << "Field ticks should be half of frame ticks for " << fps << " fps";
}

static std::string FieldTimingParamName(const testing::TestParamInfo<uint32_t>& info)
{
    return "fps_" + std::to_string(info.param);
}

INSTANTIATE_TEST_SUITE_P(
    Interlaced_FieldTimingDividedByTwo,
    AncillarySettingsCalculatorTest_FieldTimingDividedByTwo,
    testing::Values(24u, 25u, 30u, 50u, 60u),
    FieldTimingParamName);

/* Test interlaced with multiple common frame rates (parameterized) */
TEST_P(AncillarySettingsCalculatorTest_CommonFrameRates, FieldTimeAndTicksMatchExpected)
{
    const auto& param = GetParam();
    const uint32_t fps = std::get<0>(param);
    const double expected_field_time_ms = std::get<1>(param);
    const double expected_ticks = std::get<2>(param);

    init_settings(VideoScanType::Interlaced, FrameRate(fps));

    double field_time_ms = m_settings.media_unit_time_interval_ns / 1000000.0;
    EXPECT_NEAR(field_time_ms, expected_field_time_ms, 0.001)
        << "Field time mismatch for " << fps << " fps";

    EXPECT_NEAR(rational_cast<double>(m_settings.ticks_per_media_unit), expected_ticks, 1.0)
        << "Ticks mismatch for " << fps << " fps";
}

static std::string CommonFrameRatesParamName(const testing::TestParamInfo<CommonFrameRateParam>& info)
{
    return "fps_" + std::to_string(std::get<0>(info.param));
}

INSTANTIATE_TEST_SUITE_P(
    Interlaced_CommonFrameRates,
    AncillarySettingsCalculatorTest_CommonFrameRates,
    testing::Values(
        CommonFrameRateParam{24, 20.833333, 1875.0},    // 24 fps = 48 fields/sec
        CommonFrameRateParam{25, 20.0, 1800.0},         // 25 fps = 50 fields/sec
        CommonFrameRateParam{30, 16.666667, 1500.0},    // 30 fps = 60 fields/sec
        CommonFrameRateParam{50, 10.0, 900.0},          // 50 fps = 100 fields/sec
        CommonFrameRateParam{60, 8.333333, 750.0}       // 60 fps = 120 fields/sec
    ),
    CommonFrameRatesParamName);

/* Test validation failure with empty data identifiers */
TEST_F(AncillarySettingsCalculatorTest, ValidationFailure_EmptyDataIdentifiers)
{
    create_settings(VideoScanType::Interlaced, FrameRate(25));
    m_settings.data_identifiers.clear();

    EXPECT_EQ(calculate(), ReturnStatus::failure)
        << "Empty data identifiers should cause validation failure";
}

/* Test validation failure with zero max ancillary packets */
TEST_F(AncillarySettingsCalculatorTest, ValidationFailure_ZeroMaxAncillaryPackets)
{
    create_settings(VideoScanType::Interlaced, FrameRate(25));
    m_settings.max_ancillary_data_packets_per_packet = 0;

    EXPECT_EQ(calculate(), ReturnStatus::failure)
        << "Zero max_ancillary_data_packets_per_packet should cause validation failure";
}

/* Test interlaced 1080i25 (50Hz system) */
TEST_F(AncillarySettingsCalculatorTest, Interlaced_1080i25_PAL)
{
    init_settings(VideoScanType::Interlaced, FrameRate(25));

    EXPECT_NEAR(m_settings.media_unit_time_interval_ns, 20000000.0, 1.0)
        << "PAL field time should be 20ms";

    EXPECT_NEAR(rational_cast<double>(m_settings.ticks_per_media_unit), 1800.0, 1.0)
        << "PAL RTP ticks should be 1800 per field";
}

/* Test interlaced 1080i30 (60Hz system) */
TEST_F(AncillarySettingsCalculatorTest, Interlaced_1080i30_NTSC)
{
    init_settings(VideoScanType::Interlaced, FrameRate(30));

    EXPECT_NEAR(m_settings.media_unit_time_interval_ns, 16666666.67, 1.0)
        << "NTSC field time should be ~16.67ms";

    EXPECT_NEAR(rational_cast<double>(m_settings.ticks_per_media_unit), 1500.0, 1.0)
        << "NTSC RTP ticks should be 1500 per field";
}

/* Test sample rate is always 90kHz */
TEST_F(AncillarySettingsCalculatorTest, SampleRateAlways90kHz)
{
    init_settings(VideoScanType::Progressive, FrameRate(60));
    EXPECT_EQ(m_settings.sample_rate, 90000) << "Progressive sample rate must be 90kHz";

    init_settings(VideoScanType::Interlaced, FrameRate(25));
    EXPECT_EQ(m_settings.sample_rate, 90000) << "Interlaced sample rate must be 90kHz";
}
