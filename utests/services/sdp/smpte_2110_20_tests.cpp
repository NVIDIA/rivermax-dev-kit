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
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "rdk/services/media/media_defs.h"
#include "rdk/services/sdp/sdp_smpte_2110_20_description.h"
#include "rdk/services/sdp/sdp_common_descriptions.h"
#include "rdk/services/sdp/sdp_manager.h"

using namespace rivermax::dev_kit::services;
using namespace testing;

static std::string generate_complete_sdp_20(std::unique_ptr<SMPTE2110_20_MediaDescription> media_desc)
{
    auto session_desc = SessionDescription::Builder("192.168.1.100")
        .set_session_name("SMPTE ST2110-20")
        .build();
    auto time_desc = TimeDescription::Builder().build();
    auto sdp_manager = SDPManager::Builder(std::move(session_desc), std::move(time_desc))
        .add_media_description(std::move(media_desc))
        .build();
    return sdp_manager->to_string();
}

class SMPTE2110_20_MediaDescriptionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        source_filter = std::shared_ptr<SourceFilterAttribute>(
            SourceFilterAttribute::Builder("239.168.1.1", "192.168.1.100").build());
    }

    /**
     * Build an SDP string using a default FHD 1080p builder with common settings,
     * applying optional customization via a callback before building.
     */
    template <typename ConfigFunc>
    std::string build_default_fhd_sdp(ConfigFunc configure)
    {
        auto builder = SMPTE2110_20_MediaDescription::Builder(
            5000, TransportProtocol::RTP_AVP, "96", "239.168.1.1");
        builder.set_payload_type(static_cast<uint8_t>(96))
            .set_media_format(96)
            .set_sampling(VideoSampling::YCbCr_4_2_2)
            .set_depth(VideoBitDepth::_10)
            .set_width(1920)
            .set_height(1080);
        configure(builder);
        auto desc = builder.build();
        if (!desc) {
            ADD_FAILURE() << "Builder returned nullptr";
            return {};
        }
        return generate_complete_sdp_20(std::move(desc));
    }

    std::shared_ptr<SourceFilterAttribute> source_filter;
};

/* Test reference clock PTP tracable with SDP validation */
TEST_F(SMPTE2110_20_MediaDescriptionTest, TimestampRefClock_PTP_Traceable)
{
    std::string sdp = build_default_fhd_sdp([](auto& builder) {
        builder.set_timestamp_ref_clock(TimestampRefClock::PTP)
            .set_timestamp_ref_clock_ptp_traceable(true);
    });
    EXPECT_TRUE(sdp.find("a=ts-refclk:ptp=IEEE1588-2008:traceable") != std::string::npos)
        << "Missing timestamp ref clock for PTP traceable\nSDP: " << sdp;
}

/* Test reference clocks local MAC with SDP validation */
TEST_F(SMPTE2110_20_MediaDescriptionTest, TimestampRefClock_LocalMAC)
{
    std::string sdp = build_default_fhd_sdp([](auto& builder) {
        builder.set_timestamp_ref_clock(TimestampRefClock::LocalMAC)
            .set_timestamp_ref_clock_local_mac("aa-bb-cc-dd-ee-ff");
    });
    EXPECT_TRUE(sdp.find("a=ts-refclk:localmac=") != std::string::npos)
        << "Missing timestamp ref clock for Local MAC\nSDP: " << sdp;
}

/* Test PTP clock with grandmaster and domain with SDP validation */
TEST_F(SMPTE2110_20_MediaDescriptionTest, PTPClockWithGrandmasterAndDomain)
{
    std::string sdp = build_default_fhd_sdp([](auto& builder) {
        builder.set_timestamp_ref_clock(TimestampRefClock::PTP)
            .set_timestamp_ref_clock_ptp_traceable(false)
            .set_timestamp_ref_clock_ptp_grandmaster_clock_identity("00-11-22-33-44-55-66-77")
            .set_timestamp_ref_clock_ptp_domain_number(127);
    });
    EXPECT_TRUE(sdp.find("a=ts-refclk:ptp=IEEE1588-2008:00-11-22-33-44-55-66-77:127") != std::string::npos)
        << "Missing PTP grandmaster/domain configuration\nSDP: " << sdp;
}

/* Test media clock direct configurations with SDP validation */
TEST_F(SMPTE2110_20_MediaDescriptionTest, MediaClock_Direct)
{
    std::string sdp = build_default_fhd_sdp([](auto& builder) {
        builder.set_media_clock(MediaClock::Direct);
    });
    EXPECT_TRUE(sdp.find("a=mediaclk:direct=0") != std::string::npos)
        << "Missing media clock direct=0\nSDP: " << sdp;
}

/* Test media clock sender configurations with SDP validation */
TEST_F(SMPTE2110_20_MediaDescriptionTest, MediaClock_Sender)
{
    std::string sdp = build_default_fhd_sdp([](auto& builder) {
        builder.set_media_clock(MediaClock::Sender);
    });
    EXPECT_TRUE(sdp.find("a=mediaclk:sender") != std::string::npos)
        << "Missing media clock sender\nSDP: " << sdp;
}

/* Test 4K 3840x2160 production scenario */
TEST_F(SMPTE2110_20_MediaDescriptionTest, _2160)
{
    auto _2160_desc = SMPTE2110_20_MediaDescription::Builder(
        5000, TransportProtocol::RTP_AVP, "96", "239.168.100.10");
    _2160_desc.set_payload_type(static_cast<uint8_t>(96))
        .set_media_format(96)
        .set_source_filter(SourceFilterAttribute::Builder("239.168.100.10", "192.168.100.50").build())
        .set_sampling(VideoSampling::YCbCr_4_2_2)
        .set_depth(VideoBitDepth::_10)
        .set_width(_2160_WIDTH)
        .set_height(_2160_HEIGHT)
        .set_exact_frame_rate("60")
        .set_colorimetry(Colorimetry::BT2020)
        .set_packaging_mode(PackagingMode::_2110GPM)
        .set_smpte_standard_number(SMPTEStandardNumber::ST2110_20_2021)
        .set_sender_type(SenderType::_2110TPW)
        .set_timestamp_ref_clock(TimestampRefClock::PTP)
        .set_timestamp_ref_clock_ptp_traceable(true)
        .set_media_clock(MediaClock::Direct)
        .set_max_udp(8960);

    auto desc = _2160_desc.build();
    ASSERT_NE(desc, nullptr);
    std::string actual_sdp = generate_complete_sdp_20(std::move(desc));

    std::string expected_sdp =
        "v=0\r\n"
        "o=- 0 0 IN IP4 192.168.1.100\r\n"
        "s=SMPTE ST2110-20\r\n"
        "t=0 0\r\n"
        "m=video 5000 RTP/AVP 96\r\n"
        "c=IN IP4 239.168.100.10/64\r\n"
        "a=rtpmap:96 raw/90000\r\n"
        "a=fmtp:96 sampling=YCbCr-4:2:2; width=3840; height=2160; exactframerate=60; depth=10; colorimetry=BT2020; PM=2110GPM; SSN=ST2110-20:2021; TP=2110TPW; MAXUDP=8960\r\n"
        "a=source-filter: incl IN IP4 239.168.100.10 192.168.100.50\r\n"
        "a=ts-refclk:ptp=IEEE1588-2008:traceable\r\n"
        "a=mediaclk:direct=0\r\n";

    EXPECT_EQ(actual_sdp, expected_sdp) << "Complete SDP validation failed";
}

/* Test interlaced FHD video stream scenario */
TEST_F(SMPTE2110_20_MediaDescriptionTest, Interlaced)
{
    auto fhd_interlaced = SMPTE2110_20_MediaDescription::Builder(
        5002, TransportProtocol::RTP_AVP, "97", "239.168.100.11");
    fhd_interlaced.set_payload_type(static_cast<uint8_t>(97))
        .set_media_format(97)
        .set_sampling(VideoSampling::YCbCr_4_2_2)
        .set_depth(VideoBitDepth::_10)
        .set_width(1920)
        .set_height(1080)
        .set_exact_frame_rate("30")
        .set_colorimetry(Colorimetry::BT709)
        .set_video_scan_type(VideoScanType::Interlaced)
        .set_sender_type(SenderType::_2110TPN);

    auto desc = fhd_interlaced.build();
    ASSERT_NE(desc, nullptr);
    std::string actual_sdp2 = generate_complete_sdp_20(std::move(desc));

    std::string expected_sdp2 =
        "v=0\r\n"
        "o=- 0 0 IN IP4 192.168.1.100\r\n"
        "s=SMPTE ST2110-20\r\n"
        "t=0 0\r\n"
        "m=video 5002 RTP/AVP 97\r\n"
        "c=IN IP4 239.168.100.11/64\r\n"
        "a=rtpmap:97 raw/90000\r\n"
        "a=fmtp:97 sampling=YCbCr-4:2:2; width=1920; height=1080; exactframerate=30; depth=10; colorimetry=BT709; PM=2110GPM; SSN=ST2110-20:2017; TP=2110TPN; interlace\r\n"
        "a=ts-refclk:ptp=IEEE1588-2008:traceable\r\n"
        "a=mediaclk:direct=0\r\n";

    EXPECT_EQ(actual_sdp2, expected_sdp2) << "Complete SDP validation failed";
}

/* Test minimal configuration with SDP validation */
TEST_F(SMPTE2110_20_MediaDescriptionTest, MinimalConfiguration)
{
    auto builder = SMPTE2110_20_MediaDescription::Builder(
        5000, TransportProtocol::RTP_AVP, "96", "239.168.1.1");
    builder.set_payload_type(static_cast<uint8_t>(96))
        .set_media_format(96);

    auto desc = builder.build();
    ASSERT_NE(desc, nullptr);

    std::string actual_sdp = generate_complete_sdp_20(std::move(desc));
    EXPECT_FALSE(actual_sdp.empty());

    std::string expected_min_sdp =
        "v=0\r\n"
        "o=- 0 0 IN IP4 192.168.1.100\r\n"
        "s=SMPTE ST2110-20\r\n"
        "t=0 0\r\n"
        "m=video 5000 RTP/AVP 96\r\n"
        "c=IN IP4 239.168.1.1/64\r\n"
        "a=rtpmap:96 raw/90000\r\n"
        "a=fmtp:96 sampling=YCbCr-4:2:2; width=1920; height=1080; exactframerate=60; depth=10; colorimetry=BT709; PM=2110GPM; SSN=ST2110-20:2017; TP=2110TPN\r\n"
        "a=ts-refclk:ptp=IEEE1588-2008:traceable\r\n"
        "a=mediaclk:direct=0\r\n";

    EXPECT_EQ(actual_sdp, expected_min_sdp) << "Complete SDP validation failed";
}

/* Test comprehensive configuration with SDP validation */
TEST_F(SMPTE2110_20_MediaDescriptionTest, ComprehensiveConfiguration)
{
    std::vector<FormatSpecificParameter> extra_params = {
        {"broadcast-standard", "ATSC", true},
        {"region", "US", true}
    };

    auto builder = SMPTE2110_20_MediaDescription::Builder(
        5000, TransportProtocol::RTP_AVP, "96", "239.168.1.1");
    builder.set_payload_type(static_cast<uint8_t>(96))
        .set_media_format(96)
        .set_source_filter(source_filter)
        .set_sampling(VideoSampling::YCbCr_4_4_4)
        .set_depth(VideoBitDepth::_12)
        .set_width(4096)
        .set_height(2160)
        .set_exact_frame_rate("25")
        .set_colorimetry(Colorimetry::ST2065_1)
        .set_packaging_mode(PackagingMode::_2110BPM)
        .set_smpte_standard_number(SMPTEStandardNumber::ST2110_20_2021)
        .set_sender_type(SenderType::_2110TPW)
        .set_video_scan_type(VideoScanType::Progressive)
        .set_cmax(8960)
        .set_max_udp(9000)
        .set_timestamp_mode(TimestampMode::PRES)
        .set_timestamp_delay(125)
        .set_timestamp_ref_clock(TimestampRefClock::PTP)
        .set_timestamp_ref_clock_ptp_traceable(false)
        .set_timestamp_ref_clock_ptp_grandmaster_clock_identity("aa-bb-cc-dd-ee-ff-00-11")
        .set_timestamp_ref_clock_ptp_domain_number(24)
        .set_media_clock(MediaClock::Sender)
        .set_extra_format_specific_parameters(extra_params);

    auto desc = builder.build();
    ASSERT_NE(desc, nullptr);

    std::string actual_sdp = generate_complete_sdp_20(std::move(desc));

    std::string expected_sdp =
        "v=0\r\n"
        "o=- 0 0 IN IP4 192.168.1.100\r\n"
        "s=SMPTE ST2110-20\r\n"
        "t=0 0\r\n"
        "m=video 5000 RTP/AVP 96\r\n"
        "c=IN IP4 239.168.1.1/64\r\n"
        "a=rtpmap:96 raw/90000\r\n"
        "a=fmtp:96 sampling=YCbCr-4:4:4; width=4096; height=2160; exactframerate=25; depth=12; colorimetry=ST2065-1; PM=2110BPM; SSN=ST2110-20:2021; TP=2110TPW; CMAX=8960; MAXUDP=9000; TSMODE=PRES; TSDELAY=125; broadcast-standard=ATSC; region=US\r\n"
        "a=source-filter: incl IN IP4 239.168.1.1 192.168.1.100\r\n"
        "a=ts-refclk:ptp=IEEE1588-2008:aa-bb-cc-dd-ee-ff-00-11:24\r\n"
        "a=mediaclk:sender\r\n";

    EXPECT_EQ(actual_sdp, expected_sdp) << "Complete SDP validation failed";
}

/* Test extra format specific parameters with SDP validation */
TEST_F(SMPTE2110_20_MediaDescriptionTest, ExtraFormatSpecificParameters)
{
    std::vector<FormatSpecificParameter> extra_params = {
        {"custom-param1", "value1", true},
        {"custom-param2", "value2", true},
        {"broadcast-flag", "", true}  /* Parameter without value */
    };

    std::string sdp = build_default_fhd_sdp([&](auto& builder) {
        builder.set_extra_format_specific_parameters(extra_params);
    });
    EXPECT_TRUE(sdp.find("custom-param1=value1") != std::string::npos)
        << "Missing custom parameter 1\nSDP: " << sdp;
    EXPECT_TRUE(sdp.find("custom-param2=value2") != std::string::npos)
        << "Missing custom parameter 2\nSDP: " << sdp;
    EXPECT_TRUE(sdp.find("broadcast-flag") != std::string::npos)
        << "Missing broadcast flag parameter\nSDP: " << sdp;
}

/* Test interlaced parameter present for 1080i50 */
TEST_F(SMPTE2110_20_MediaDescriptionTest, Interlaced_1080i50_ContainsInterlaceParameter)
{
    std::string sdp = build_default_fhd_sdp([](auto& builder) {
        builder.set_exact_frame_rate("25")
            .set_video_scan_type(VideoScanType::Interlaced);
    });
    EXPECT_TRUE(sdp.find("interlace") != std::string::npos)
        << "Missing 'interlace' parameter for 1080i50\nSDP: " << sdp;
    EXPECT_TRUE(sdp.find("exactframerate=25") != std::string::npos)
        << "Missing frame rate for 1080i50 (25 fps = 50 fields/sec)\nSDP: " << sdp;
}

/* Test interlaced parameter present for 1080i60 */
TEST_F(SMPTE2110_20_MediaDescriptionTest, Interlaced_1080i60_ContainsInterlaceParameter)
{
    std::string sdp = build_default_fhd_sdp([](auto& builder) {
        builder.set_exact_frame_rate("30000/1001")
            .set_video_scan_type(VideoScanType::Interlaced);
    });
    EXPECT_TRUE(sdp.find("interlace") != std::string::npos)
        << "Missing 'interlace' parameter for 1080i60\nSDP: " << sdp;
    EXPECT_TRUE(sdp.find("exactframerate=30000/1001") != std::string::npos)
        << "Missing fractional frame rate for 1080i60\nSDP: " << sdp;
}

/* Test interlaced parameter NOT present for progressive video */
TEST_F(SMPTE2110_20_MediaDescriptionTest, Progressive_DoesNotContainInterlaceParameter)
{
    std::string sdp = build_default_fhd_sdp([](auto& builder) {
        builder.set_exact_frame_rate("60")
            .set_video_scan_type(VideoScanType::Progressive);
    });
    EXPECT_TRUE(sdp.find("interlace") == std::string::npos)
        << "Progressive video should NOT contain 'interlace' parameter\nSDP: " << sdp;
}
