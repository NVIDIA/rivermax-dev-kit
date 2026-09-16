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

/* Platform specific headers and declarations */
#if defined(_WIN32)
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#elif defined(__linux__)
#include <arpa/inet.h>
#endif

#include "rdk/services/protocol/media_packet_parser.h"

namespace rdk {
namespace services {

bool MediaPacketParser::get_sequence_number(const byte_t* data, size_t length, bool is_extended, uint32_t& sequence_number) const
{
    if (unlikely(!data || length < m_rtp_offset || length - m_rtp_offset < sizeof(protocol::RTPHeader))) {
        return false;
    }

    const protocol::RTPHeader* rtp_header = rtp(data);
    if (unlikely(!rtp_header)) {
        return false;
    }

    uint32_t seq_num = ntohs(rtp_header->sequence_number);
    if (is_extended) {
        uint16_t esn;
        if (unlikely(!get_extended_sequence_number_from_rtp(rtp_header, length - m_rtp_offset, esn))) {
            return false;
        }
        seq_num |= (static_cast<uint32_t>(esn) << 16);
    }

    sequence_number = seq_num;
    return true;
}

bool MediaPacketParser::get_extended_sequence_number_from_rtp(const protocol::RTPHeader* rtp_header, size_t rtp_length, uint16_t& ext_seq_num) const
{
    const uint8_t csrc_count = rtp_header->vpxcc & 0x0F;
    const size_t esn_offset = sizeof(protocol::RTPHeader) + csrc_count * 4;

    if (unlikely(esn_offset + sizeof(uint16_t) > rtp_length)) {
        return false;
    }

    const byte_t* esn_ptr = reinterpret_cast<const byte_t*>(rtp_header) + esn_offset;
    ext_seq_num = (static_cast<uint16_t>(esn_ptr[0]) << 8) | esn_ptr[1];
    return true;
}

} // namespace services
} // namespace rdk
