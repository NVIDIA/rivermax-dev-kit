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

#include "rdk/services/sdp/sdp_smpte_2110_40_description.h"
#include "rdk/services/sdp/sdp_common_descriptions.h"
#include "rdk/services/sdp/sdp_manager.h"

using namespace rdk::services;
using namespace testing;

static std::string generate_complete_sdp_40(std::unique_ptr<SMPTE2110_40_MediaDescription> media_desc) {
    auto session_desc = SessionDescription::Builder("192.168.1.100")
        .set_session_name("SMPTE ST2110-40")
        .build();
    auto time_desc = TimeDescription::Builder().build();
    auto sdp_manager = SDPManager::Builder(std::move(session_desc), std::move(time_desc))
        .add_media_description(std::move(media_desc))
        .build();
    return sdp_manager->to_string();
}

static bool sdp_contains_string(const std::string& sdp, const std::string& expected_string) {
    return sdp.find(expected_string) != std::string::npos;
}

class SMPTE2110_40_MediaDescriptionTest : public ::testing::Test {};

/* Test minimal configuration with SDP validation */
TEST_F(SMPTE2110_40_MediaDescriptionTest, MinimalConfiguration) {
    auto anc_desc = SMPTE2110_40_MediaDescription::Builder(
        5008, TransportProtocol::RTP_AVP, "100", "239.168.1.1")
        .build();

    ASSERT_NE(anc_desc, nullptr);

    std::string actual_sdp = generate_complete_sdp_40(std::move(anc_desc));
    EXPECT_FALSE(actual_sdp.empty());

    std::string expected_sdp =
        "v=0\r\n"
        "o=- 0 0 IN IP4 192.168.1.100\r\n"
        "s=SMPTE ST2110-40\r\n"
        "t=0 0\r\n"
        "m=video 5008 RTP/AVP 100\r\n"
        "c=IN IP4 239.168.1.1/64\r\n"
        "a=rtpmap:100 smpte291/90000\r\n"
        "a=fmtp:100 \r\n"
        "a=ts-refclk:ptp=IEEE1588-2008:traceable\r\n"
        "a=mediaclk:direct=0\r\n";

    EXPECT_EQ(actual_sdp, expected_sdp);
}

/* DID_SDID with VPID_Code */
TEST_F(SMPTE2110_40_MediaDescriptionTest, DIDSDID_With_VPIDCode) {
    auto anc_desc = SMPTE2110_40_MediaDescription::Builder(
        5008, TransportProtocol::RTP_AVP, "97", "239.168.1.1")
        .set_did_sdid(0x41, 0xA5)
        .set_vpid_code(static_cast<uint8_t>(132))
        .build();

    ASSERT_NE(anc_desc, nullptr);

    std::string actual_sdp = generate_complete_sdp_40(std::move(anc_desc));
    EXPECT_FALSE(actual_sdp.empty());

    std::string expected_sdp =
        "v=0\r\n"
        "o=- 0 0 IN IP4 192.168.1.100\r\n"
        "s=SMPTE ST2110-40\r\n"
        "t=0 0\r\n"
        "m=video 5008 RTP/AVP 97\r\n"
        "c=IN IP4 239.168.1.1/64\r\n"
        "a=rtpmap:97 smpte291/90000\r\n"
        "a=fmtp:97 DID_SDID={0x41,0xA5}; VPID_Code=132\r\n"
        "a=ts-refclk:ptp=IEEE1588-2008:traceable\r\n"
        "a=mediaclk:direct=0\r\n";

    EXPECT_EQ(actual_sdp, expected_sdp);
}

/* Double DID_SDID and change in payload type and media format */
TEST_F(SMPTE2110_40_MediaDescriptionTest, DoubleDIDSDID) {
    auto anc_desc = SMPTE2110_40_MediaDescription::Builder(
        5008, TransportProtocol::RTP_AVP, "97", "239.168.1.1")
        .set_did_sdid(0x61, 0x02)
        .add_did_sdid(0x41, 0x05)
        .set_payload_type(110)
        .set_media_format(111)
        .build();

    ASSERT_NE(anc_desc, nullptr);

    std::string actual_sdp = generate_complete_sdp_40(std::move(anc_desc));
    EXPECT_FALSE(actual_sdp.empty());

    std::string expected_sdp =
        "v=0\r\n"
        "o=- 0 0 IN IP4 192.168.1.100\r\n"
        "s=SMPTE ST2110-40\r\n"
        "t=0 0\r\n"
        "m=video 5008 RTP/AVP 97\r\n"
        "c=IN IP4 239.168.1.1/64\r\n"
        "a=rtpmap:110 smpte291/90000\r\n"
        "a=fmtp:111 DID_SDID={0x61,0x02}; DID_SDID={0x41,0x05}\r\n"
        "a=ts-refclk:ptp=IEEE1588-2008:traceable\r\n"
        "a=mediaclk:direct=0\r\n";

    EXPECT_EQ(actual_sdp, expected_sdp);
}

/* Test extra format specific parameters with SDP validation */
TEST_F(SMPTE2110_40_MediaDescriptionTest, ExtraFormatSpecificParameters) {
    std::vector<FormatSpecificParameter> extra_params = {
        {"custom-param1", "value1", true},
        {"custom-param2", "value2", true},
        {"broadcast-flag", "", true}
    };

    auto anc_desc = SMPTE2110_40_MediaDescription::Builder(
        5008, TransportProtocol::RTP_AVP, "100", "239.168.1.1")
        .set_extra_format_specific_parameters(extra_params)
        .build();

    ASSERT_NE(anc_desc, nullptr);

    std::string sdp = generate_complete_sdp_40(std::move(anc_desc));
    EXPECT_TRUE(sdp_contains_string(sdp, "custom-param1=value1"));
    EXPECT_TRUE(sdp_contains_string(sdp, "custom-param2=value2"));
    EXPECT_TRUE(sdp_contains_string(sdp, "broadcast-flag"));
}
