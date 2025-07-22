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

#ifndef RDK_SERVICES_PROTOCOL_UDP_H_
#define RDK_SERVICES_PROTOCOL_UDP_H_

#include <cstdint>

namespace rivermax
{
namespace dev_kit
{
namespace services
{

#pragma pack(push, 1)

/**
 * @brief: UDP header (RFC 768).
 *
 * Provides a packed representation of the 8-byte UDP segment header. All
 * 16-bit members are stored in <b>network byte order</b> (big-endian).
 */
struct UDPHeader
{
    /** Source port. */
    uint16_t src_port;
    /** Destination port. */
    uint16_t dst_port;
    /** Length of UDP datagram in bytes (header + payload). */
    uint16_t length;
    /** Checksum (0 indicates none for IPv4). */
    uint16_t checksum;
};

#pragma pack(pop)

// ============================================================================
// UDP PROTOCOL CONSTANTS
// ============================================================================

/** UDP header size in bytes (always 8). */
constexpr size_t UDP_HEADER_SIZE = 8;

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif // RDK_SERVICES_PROTOCOL_UDP_H_