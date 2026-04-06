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
    const protocol::RTPHeader* rtp_header = rtp(data);
    if (!rtp_header) {
        return false;
    }

    sequence_number = ntohs(rtp_header->sequence_number);
    if (is_extended) {
        uint16_t esn = get_extended_sequence_number_from_rtp(rtp_header, data, length);
        sequence_number |= (static_cast<uint32_t>(esn) << 16);
    }

    return true;
}

uint16_t MediaPacketParser::get_extended_sequence_number_from_rtp(const protocol::RTPHeader* rtp_header, const byte_t* data, size_t length) const
{
    uint8_t csrc_count = rtp_header->vpxcc & 0x0F;
    const byte_t* esn_ptr = reinterpret_cast<const byte_t*>(rtp_header) +
                           sizeof(protocol::RTPHeader) +
                           csrc_count * 4;

    if (esn_ptr + 2 > data + length) {
        return 0;
    }
    return ntohs(*reinterpret_cast<const uint16_t*>(esn_ptr));
}

} // namespace services
} // namespace rdk
