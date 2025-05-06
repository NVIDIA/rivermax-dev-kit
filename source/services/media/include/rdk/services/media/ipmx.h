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

#ifndef RDK_SERVICES_MEDIA_IPMX_H_
#define RDK_SERVICES_MEDIA_IPMX_H_

#include <cstddef>
#include <cstdint>

#include "rdk/services/utils/defs.h"
#include "rdk/services/error_handling/return_status.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
static constexpr uint8_t RTP_VERSION = 2;
static constexpr uint8_t RTCP_SOURCE_CNT = 1;
static constexpr uint16_t IPMX_TAG = 0x5831;
static constexpr uint16_t IPMX_MIB_TYPE_UNCOMPRESSED_VIDEO = 0x0001;
/**
 * @brief: RTCP packets types accepted in the RTCP compound packet that contains an
 * IPMX Sender Report.
 */
enum class RTCPPacketType {
    SenderReport = 200,
    SourceDescr = 202
};
/**
 * @brief: Acceptable item IDs for SDES packet in the IPMX Sender Report compound.
 */
enum class RTCPSdesType {
    Cname = 1,
    End = 0
};
/**
 * @brief: IPMX Info block of the IPMX Sender Report packet.
 */
struct IPMXInfoBlock {
    uint16_t ipmx_tag;
    uint16_t length;
    uint8_t block_version;
    uint8_t pad[3];
    uint8_t ts_refclk[64];
    uint8_t mediaclk[12];
};
/**
 * @brief: Helper function to encode a media_bits/interlaced half-word in IPMX Sender Report.
 */
static inline uint16_t pack_media_bits_interlace(uint8_t bits, bool is_interlaced)
{
    return ((static_cast<uint16_t>(bits) & 0x7f) << 8) | ((!!(is_interlaced)) << 5);
}
/**
 * @brief: IPMX Info block of the IPMX Sender Report packet.
 */
struct IPMXMediaInfoBlock {
    uint16_t type;
    uint16_t length;
    uint8_t sampling[16];
    uint16_t packing;
    uint8_t par_w;
    uint8_t par_h;
    uint8_t range[12];
    uint8_t colorimetry[20];
    uint8_t tcs[16];
    uint16_t width;
    uint16_t height;
    uint32_t rate;
    uint32_t pixel_clk_hi;
    uint32_t pixel_clk_lo;
    uint16_t htotal;
    uint16_t vtotal;
};
/**
 * @brief: IPMX Sender Report packet.
 */
struct IPMXSenderReport {
    uint16_t type_ver;
    uint16_t length;
    uint32_t ssrc;
    uint32_t ntp_ts_hi;
    uint32_t ntp_ts_lo;
    uint32_t rtp_ts;
    uint32_t pkt_cnt;
    uint32_t byte_cnt;
    IPMXInfoBlock info;
    IPMXMediaInfoBlock media;
};
/**
 * @brief: SDES (Source Description) packet parameters of the RTCP compound.
 */
static constexpr uint8_t RTCP_SDES_CNAME_LEN_MAX = 255;
static constexpr uint8_t RTCP_SDES_PAD_LEN_MAX = 3;
/**
 * @brief: CNAME item of SDES packet.
 */
struct RTCPSdesCname {
    uint8_t type;
    uint8_t length;
};
/**
 * @brief: END item of SDES packet.
 */
struct RTCPSdesEnd {
    uint8_t type;
};
/**
 * @brief: SDES chunk header.
 */
struct RTCPSdesChunk {
    uint32_t ssrc;
};
/**
 * @brief: SDES packet header.
 */
struct RTCPSourceDescr {
    uint16_t type_ver;
    uint16_t length;
};
/**
 * @brief: SDES packet amximal size.
 */
static constexpr size_t RTCP_SDES_SIZE_MAX = sizeof(RTCPSourceDescr) +
                                             sizeof(RTCPSdesChunk) +
                                             sizeof(RTCPSdesCname) + RTCP_SDES_CNAME_LEN_MAX +
                                             sizeof(RTCPSdesEnd) + RTCP_SDES_PAD_LEN_MAX;
/**
 * @brief: RTCP compound packet containing IPMX Sender Report and SDES packet.
 */
struct RTCPCompoundPacket {
    IPMXSenderReport sr;
    uint8_t sdes_raw[RTCP_SDES_SIZE_MAX];
};
/**
 * @brief: Video stream parameters comprising the IPMX Info block of the IPMX Sender Report packet
 */
struct IPMXStreamVideoInfo {
    char sampling_format[17];
    uint8_t bit_depth;
    bool is_floating;
    bool is_gpm;
    bool is_interlaced;
    bool is_segmented;
    uint8_t par_w;
    uint8_t par_h;
    char range[13];
    char colorimetry[21];
    char tcs[17];
    uint16_t width;
    uint16_t height;
    uint32_t rate_num;
    uint8_t rate_denom;
    uint32_t pixel_clock;
    uint16_t h_total;
    uint16_t v_total;
};
/**
 * @brief: Contents of IPMX Sender Report RTCP packet (IPMX Sender Report and SDES packet)
 */
struct IPMXStreamInfo {
    uint8_t rtp_version;
    uint32_t ssrc;
    uint64_t ntp_ts;
    uint32_t rtp_ts;
    uint32_t pkt_cnt;
    uint32_t byte_cnt;
    uint8_t info_block_version;
    char ts_refclk[65];
    char mediaclk[13];
    uint8_t media_block_type;
    union {
        IPMXStreamVideoInfo video;
    } media;
    char cname[256];
    /**
    * @brief: Fills the member fields by parsing the IPMX Sender Report RTCP packet.
    *
    * @param [in] data: Pointer to the packet data.
    * @param [in] info: Reference to the number of consumed packets.
    *
    * @return: Status of the operation.
    */
    ReturnStatus fill_from_rtcp_sr_packet(const byte_t* data, size_t len);
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_IPMX_H_ */
