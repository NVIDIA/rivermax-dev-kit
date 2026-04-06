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

#include "rdk/services/sdp/sdp_smpte_2110_30_description.h"
#include "rdk/services/sdp/sdp_common_descriptions.h"
#include "rdk/services/sdp/sdp_manager.h"

using namespace rdk::services;
using namespace testing;

static std::string generate_sdp_2110_30(std::unique_ptr<SMPTE2110_30_MediaDescription> media_desc) {
    auto session_desc = SessionDescription::Builder("192.168.1.100")
        .set_session_name("SMPTE ST2110-30")
        .build();
    auto time_desc = TimeDescription::Builder().build();
    auto sdp_manager = SDPManager::Builder(std::move(session_desc), std::move(time_desc))
        .add_media_description(std::move(media_desc))
        .build();
    return sdp_manager->to_string();
}

class SMPTE2110_30_MediaDescriptionTest : public ::testing::Test {
protected:
    void SetUp() override {
        source_filter = std::shared_ptr<SourceFilterAttribute>(
            SourceFilterAttribute::Builder("239.168.1.1", "192.168.1.100").build());
    }

    std::shared_ptr<SourceFilterAttribute> source_filter;
};

/* Test minimal configuration with SDP validation */
TEST_F(SMPTE2110_30_MediaDescriptionTest, MinimalConfiguration) {
    auto audio_desc = SMPTE2110_30_MediaDescription::Builder(
        5004, TransportProtocol::RTP_AVP, "97", "239.168.1.1")
        .build();

    ASSERT_NE(audio_desc, nullptr);

    std::string actual_sdp = generate_sdp_2110_30(std::move(audio_desc));
    EXPECT_FALSE(actual_sdp.empty());

    std::string expected_sdp =
        "v=0\r\n"
        "o=- 0 0 IN IP4 192.168.1.100\r\n"
        "s=SMPTE ST2110-30\r\n"
        "t=0 0\r\n"
        "m=audio 5004 RTP/AVP 97\r\n"
        "c=IN IP4 239.168.1.1/64\r\n"
        "a=rtpmap:97 L24/48000/2\r\n"
        "a=fmtp:97 \r\n"
        "a=ptime:1\r\n"
        "a=maxptime:1\r\n"
        "a=ts-refclk:ptp=IEEE1588-2008:traceable\r\n"
        "a=mediaclk:direct=0\r\n";

    EXPECT_EQ(actual_sdp, expected_sdp);
}

/* Audio stream: 1ms, 48kHz, 2ch, SMPTE.(51) */
TEST_F(SMPTE2110_30_MediaDescriptionTest, ExactSample_51_1ms_48k_2ch)
{
    auto audio_desc = SMPTE2110_30_MediaDescription::Builder(
            2000, TransportProtocol::RTP_AVP, "97", "224.1.1.1")
        .set_source_filter(SourceFilterAttribute::Builder("224.1.1.1", "192.168.1.2").build())
        .set_encoding(AudioEncoding::L24)
        .set_sampling_rate(AudioSamplingRate::_48000)
        .set_channels(2)
        .set_channel_order({ChannelGroupingSymbol::_51})
        .set_ptime(1.0)
        .set_maxptime(1.0)
        .set_timestamp_ref_clock(TimestampRefClock::LocalMAC)
        .set_timestamp_ref_clock_local_mac("40-a3-6b-a0-2b-d2")
        .set_media_clock(MediaClock::Direct)
        .build();

    std::string actual_sdp = generate_sdp_2110_30(std::move(audio_desc));

    std::string expected_sdp =
        "v=0\r\n"
        "o=- 0 0 IN IP4 192.168.1.100\r\n"
        "s=SMPTE ST2110-30\r\n"
        "t=0 0\r\n"
        "m=audio 2000 RTP/AVP 97\r\n"
        "c=IN IP4 224.1.1.1/64\r\n"
        "a=rtpmap:97 L24/48000/2\r\n"
        "a=fmtp:97 channel-order=SMPTE2110.(51)\r\n"
        "a=ptime:1\r\n"
        "a=maxptime:1\r\n"
        "a=source-filter: incl IN IP4 224.1.1.1 192.168.1.2\r\n"
        "a=ts-refclk:localmac=40-a3-6b-a0-2b-d2\r\n"
        "a=mediaclk:direct=0\r\n";

    EXPECT_EQ(actual_sdp, expected_sdp);
}

/* Audio stream: 0.125ms, 96kHz, 2ch, SMPTE.(ST) */
TEST_F(SMPTE2110_30_MediaDescriptionTest, ExactSample_ST_0125ms_96k_2ch)
{
    auto audio_desc = SMPTE2110_30_MediaDescription::Builder(
            2000, TransportProtocol::RTP_AVP, "97", "224.1.1.1")
        .set_source_filter(SourceFilterAttribute::Builder("224.1.1.1", "192.168.1.2").build())
        .set_encoding(AudioEncoding::L24)
        .set_sampling_rate(AudioSamplingRate::_96000)
        .set_channels(2)
        .set_channel_order({ChannelGroupingSymbol::ST})
        .set_ptime(0.125)
        .set_maxptime(1.0)
        .set_timestamp_ref_clock(TimestampRefClock::LocalMAC)
        .set_timestamp_ref_clock_local_mac("40-a3-6b-a0-2b-d2")
        .set_media_clock(MediaClock::Direct)
        .build();

    std::string actual_sdp = generate_sdp_2110_30(std::move(audio_desc));

    std::string expected_sdp =
        "v=0\r\n"
        "o=- 0 0 IN IP4 192.168.1.100\r\n"
        "s=SMPTE ST2110-30\r\n"
        "t=0 0\r\n"
        "m=audio 2000 RTP/AVP 97\r\n"
        "c=IN IP4 224.1.1.1/64\r\n"
        "a=rtpmap:97 L24/96000/2\r\n"
        "a=fmtp:97 channel-order=SMPTE2110.(ST)\r\n"
        "a=ptime:0.125\r\n"
        "a=maxptime:1\r\n"
        "a=source-filter: incl IN IP4 224.1.1.1 192.168.1.2\r\n"
        "a=ts-refclk:localmac=40-a3-6b-a0-2b-d2\r\n"
        "a=mediaclk:direct=0\r\n";

    EXPECT_EQ(actual_sdp, expected_sdp);
}

/* Audio stream: validate mixed aliases and undefined token (Standard_Stereo, 51, U13) */
TEST_F(SMPTE2110_30_MediaDescriptionTest, ChannelOrder_Alias_Mixed_With_Undef_Rendered)
{
    auto audio_desc = SMPTE2110_30_MediaDescription::Builder(
            2000, TransportProtocol::RTP_AVP, "97", "224.1.1.1")
        .set_source_filter(SourceFilterAttribute::Builder("224.1.1.1", "192.168.1.2").build())
        .set_encoding(AudioEncoding::L24)
        .set_sampling_rate(AudioSamplingRate::_96000)
        .set_channels(2)
        .set_channel_order({ChannelGroupingSymbol::Standard_Stereo, ChannelGroupingSymbol::_51, ChannelGroupingSymbol::U01, ChannelGroupingSymbol::U10})
        .set_ptime(0.125)
        .set_maxptime(1.0)
        .set_timestamp_ref_clock(TimestampRefClock::LocalMAC)
        .set_timestamp_ref_clock_local_mac("40-a3-6b-a0-2b-d2")
        .set_media_clock(MediaClock::Direct)
        .build();

    std::string actual_sdp = generate_sdp_2110_30(std::move(audio_desc));

    std::string expected_sdp =
        "v=0\r\n"
        "o=- 0 0 IN IP4 192.168.1.100\r\n"
        "s=SMPTE ST2110-30\r\n"
        "t=0 0\r\n"
        "m=audio 2000 RTP/AVP 97\r\n"
        "c=IN IP4 224.1.1.1/64\r\n"
        "a=rtpmap:97 L24/96000/2\r\n"
        "a=fmtp:97 channel-order=SMPTE2110.(ST,51,U01,U10)\r\n"
        "a=ptime:0.125\r\n"
        "a=maxptime:1\r\n"
        "a=source-filter: incl IN IP4 224.1.1.1 192.168.1.2\r\n"
        "a=ts-refclk:localmac=40-a3-6b-a0-2b-d2\r\n"
        "a=mediaclk:direct=0\r\n";

    EXPECT_EQ(actual_sdp, expected_sdp);
}

/* Test extra format specific parameters with SDP validation plus payload type and media format change */
TEST_F(SMPTE2110_30_MediaDescriptionTest, ExtraFormatSpecificParameters) {
    std::vector<FormatSpecificParameter> extra_params = {};

    auto audio_desc = SMPTE2110_30_MediaDescription::Builder(
        5004, TransportProtocol::RTP_AVP, "97", "239.168.1.1")
        .set_payload_type(100)
        .set_media_format(101)
        .set_encoding(AudioEncoding::L24)
        .set_sampling_rate(AudioSamplingRate::_48000)
        .set_channels(2)
        .set_extra_format_specific_parameters(extra_params)
        .build();

    ASSERT_NE(audio_desc, nullptr);

    std::string actual_sdp = generate_sdp_2110_30(std::move(audio_desc));

    std::string expected_sdp =
        "v=0\r\n"
        "o=- 0 0 IN IP4 192.168.1.100\r\n"
        "s=SMPTE ST2110-30\r\n"
        "t=0 0\r\n"
        "m=audio 5004 RTP/AVP 97\r\n"
        "c=IN IP4 239.168.1.1/64\r\n"
        "a=rtpmap:100 L24/48000/2\r\n"
        "a=fmtp:101 \r\n"
        "a=ptime:1\r\n"
        "a=maxptime:1\r\n"
        "a=ts-refclk:ptp=IEEE1588-2008:traceable\r\n"
        "a=mediaclk:direct=0\r\n";

    EXPECT_EQ(actual_sdp, expected_sdp);
}
