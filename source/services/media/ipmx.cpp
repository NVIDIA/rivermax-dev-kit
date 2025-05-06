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

#include <cstring>
#include <stdexcept>
#include <string>

/* Platform specific headers and declarations */
#if defined(_WIN32)
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#elif defined(__linux__)
#include <arpa/inet.h>
#endif

#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/media/media_defs.h"
#include "rdk/services/media/ipmx.h"

using namespace rivermax::dev_kit::services;

ReturnStatus IPMXStreamInfo::fill_from_rtcp_sr_packet(const byte_t* data, size_t len)
{
    RTCPCompoundPacket pkt;
    if (len > sizeof(pkt)) {
        std::cerr << "Report packet is too long: " << std::to_string(len)
                  <<" (" + std::to_string(sizeof(pkt)) << ")" << std::endl;;
        return ReturnStatus::failure;
    }
    std::memcpy(&pkt, data, sizeof(pkt));

    rtp_version = (ntohs(pkt.sr.type_ver) >> 14) & 0x03;
    ssrc = ntohl(pkt.sr.ssrc);
    ntp_ts = static_cast<uint64_t>(ntohl(pkt.sr.ntp_ts_hi)) * NS_IN_SEC + ntohl(pkt.sr.ntp_ts_lo);
    rtp_ts = ntohl(pkt.sr.rtp_ts);
    pkt_cnt = ntohl(pkt.sr.pkt_cnt);
    byte_cnt = ntohl(pkt.sr.byte_cnt);
    info_block_version = pkt.sr.info.block_version;

    std::strncpy(ts_refclk, reinterpret_cast<char*>(pkt.sr.info.ts_refclk), sizeof(ts_refclk));
    ts_refclk[sizeof(ts_refclk) - 1] = '\0';

    std::strncpy(mediaclk, reinterpret_cast<char*>(pkt.sr.info.mediaclk), sizeof(mediaclk));
    mediaclk[sizeof(mediaclk) - 1] = '\0';

    media_block_type = ntohs(pkt.sr.media.type);
    if (media_block_type != IPMX_MIB_TYPE_UNCOMPRESSED_VIDEO) {
        std::cerr << "Unknown media block type: " << std::to_string(media_block_type) << std::endl;
        return ReturnStatus::success;
    }
    std::strncpy(media.video.sampling_format, reinterpret_cast<char*>(pkt.sr.media.sampling),
                 sizeof(media.video.sampling_format));
    media.video.sampling_format[sizeof(media.video.sampling_format) - 1] = '\0';

    media.video.is_floating = !!(ntohs(pkt.sr.media.packing) & (1 << 15));
    media.video.bit_depth = (ntohs(pkt.sr.media.packing) >> 8) & 0x7f;
    media.video.is_gpm = !!(ntohs(pkt.sr.media.packing) & (1 << 7));
    media.video.is_interlaced = !!(ntohs(pkt.sr.media.packing) & (1 << 6));
    media.video.is_segmented = !!(ntohs(pkt.sr.media.packing) & (1 << 5));
    media.video.par_w = pkt.sr.media.par_w;
    media.video.par_h = pkt.sr.media.par_h;
    std::strncpy(media.video.range, reinterpret_cast<char*>(pkt.sr.media.range),
                 sizeof(media.video.range));
    media.video.range[sizeof(media.video.range) - 1] = '\0';

    std::strncpy(media.video.colorimetry, reinterpret_cast<char*>(pkt.sr.media.colorimetry),
                 sizeof(media.video.colorimetry));
    media.video.tcs[sizeof(media.video.tcs) - 1] = '\0';

    std::strncpy(media.video.tcs, reinterpret_cast<char*>(pkt.sr.media.tcs),
                 sizeof(media.video.tcs));
    media.video.tcs[sizeof(media.video.tcs) - 1] = '\0';

    media.video.width = ntohs(pkt.sr.media.width);
    media.video.height = ntohs(pkt.sr.media.height);
    media.video.rate_num = (ntohl(pkt.sr.media.rate) >> 10) & 0x3fffff;
    media.video.rate_denom = htonl(pkt.sr.media.rate) & 0x3ff;
    media.video.pixel_clock = static_cast<uint64_t>(ntohl(pkt.sr.media.pixel_clk_hi)) << 32 |
                                                    ntohl(pkt.sr.media.pixel_clk_lo);
    media.video.h_total = htons(pkt.sr.media.htotal);
    media.video.v_total = htons(pkt.sr.media.vtotal);

    uint8_t* ptr = pkt.sdes_raw;
    ptr += sizeof(RTCPSourceDescr);
    ptr += sizeof(RTCPSdesChunk);
    RTCPSdesCname* sdes_cname_item = new (ptr) RTCPSdesCname;
    size_t cname_len = sdes_cname_item->length;
    ptr += sizeof(RTCPSdesCname);
    std::strncpy(cname, reinterpret_cast<char*>(ptr), cname_len);
    cname[cname_len] = '\0';
    return ReturnStatus::success;
}

