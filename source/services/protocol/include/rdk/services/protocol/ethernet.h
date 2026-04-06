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

#ifndef RDK_SERVICES_PROTOCOL_ETHERNET_H_
#define RDK_SERVICES_PROTOCOL_ETHERNET_H_

#include <cstdint>

namespace rdk
{
namespace services
{

#pragma pack(push, 1)

/**
 * @brief: Ethernet (IEEE 802.3) frame header.
 *
 * Represents the first 14 bytes of an Ethernet frame as it appears on the
 * wire. All multi-byte members are stored in <b>network byte order</b>
 * (big-endian).
 */
struct EthernetHeader
{
    /** Destination MAC address (6 bytes). */
    uint8_t dst_mac[6];
    /** Source MAC address (6 bytes). */
    uint8_t src_mac[6];
    /** EtherType field. */
    uint16_t ether_type;
};

#pragma pack(pop)

// ============================================================================
// ETHERNET PROTOCOL CONSTANTS
// ============================================================================

/** Ethernet header size in bytes (always 14). */
constexpr size_t ETHERNET_HEADER_SIZE = 14;
/** VLAN tag size in bytes. */
constexpr size_t VLAN_TAG_SIZE = 4;
/** EtherType for 802.1Q VLAN tagging. */
constexpr uint16_t ETH_TYPE_802_1Q = 0x8100;
/** EtherType for IPv4. */
constexpr uint16_t ETH_TYPE_IPV4 = 0x0800;
/** EtherType for IPv6. */
constexpr uint16_t ETH_TYPE_IPV6 = 0x86DD;

} // namespace services
} // namespace rdk

#endif // RDK_SERVICES_PROTOCOL_ETHERNET_H_