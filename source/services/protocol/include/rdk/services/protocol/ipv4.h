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

#ifndef RDK_SERVICES_PROTOCOL_IPV4_H_
#define RDK_SERVICES_PROTOCOL_IPV4_H_

#include <cstdint>

namespace rdk
{
namespace services
{

#pragma pack(push, 1)

/**
 * @brief: IPv4 header structure (RFC 791).
 *
 * This is a byte-accurate representation of the on-wire IPv4 header.
 * All multi-byte integer members are stored in <b>network byte order</b>
 * (big-endian).  Users must apply <code>ntohs</code>/<code>ntohl</code>
 * before inspecting the values on little-endian systems.
 */
struct IPv4Header
{
    /** Version (upper 4 bits) and Internet Header Length, IHL (lower 4 bits).
     *  Together they describe the IP version (always 4 here) and the length
     *  of the header in 32-bit words. */
    uint8_t version_ihl;
    /** Differentiated Services Code Point (DSCP, 6 bits) and Explicit
     *  Congestion Notification (ECN, 2 bits). */
    uint8_t dscp_ecn;
    /** Total length of the IPv4 packet (header + payload) in bytes. */
    uint16_t total_length;
    /** Identification field used for reassembly of fragmented packets. */
    uint16_t identification;
    /** Flags (upper 3 bits) and Fragment Offset (lower 13 bits). */
    uint16_t flags_fragment_offset;
    /** Time-to-live. */
    uint8_t ttl;
    /** Protocol. */
    uint8_t protocol;
    /** Header checksum. */
    uint16_t header_checksum;
    /** Source IP address. */
    uint32_t src_ip;
    /** Destination IP address. */
    uint32_t dst_ip;
};

#pragma pack(pop)

// ============================================================================
// IPv4 PROTOCOL CONSTANTS
// ============================================================================

/** IPv4 minimum header size in bytes (without options). */
constexpr size_t IPV4_MIN_HEADER_SIZE = 20;
/** IPv4 maximum header size in bytes (with maximum options). */
constexpr size_t IPV4_MAX_HEADER_SIZE = 60;
/** IPv4 version number (always 4). */
constexpr uint8_t IPV4_VERSION = 4;
/** IPv4 minimum Internet Header Length (IHL) value (20 bytes). */
constexpr uint8_t IPV4_MIN_IHL = 5;
/** IPv4 maximum Internet Header Length (IHL) value (60 bytes). */
constexpr uint8_t IPV4_MAX_IHL = 15;
/** IPv4 IHL multiplier - IHL is specified in 4-byte words. */
constexpr uint8_t IPV4_IHL_MULTIPLIER = 4;

} // namespace services
} // namespace rdk

#endif // RDK_SERVICES_PROTOCOL_IPV4_H_
