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

#ifndef RDK_SERVICES_PROTOCOL_MEDIA_PACKET_PARSER_H_
#define RDK_SERVICES_PROTOCOL_MEDIA_PACKET_PARSER_H_

#include <type_traits>
#include <functional>

#include "rdk/services/protocol/packet_parser.h"
#include "rdk/services/protocol/ethernet.h"
#include "rdk/services/protocol/ipv4.h"
#include "rdk/services/protocol/udp.h"
#include "rdk/services/protocol/rtp.h"
#include "rdk/services/protocol/ethernet_parser.h"
#include "rdk/services/protocol/ipv4_parser.h"
#include "rdk/services/protocol/udp_parser.h"
#include "rdk/services/protocol/rtp_parser.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 * @brief: Specialized packet parser preconfigured for media streaming protocols.
 *
 * This class extends @ref PacketParser with high-performance access to common
 * media streaming protocol headers: Ethernet → IPv4 → UDP → RTP.
 *
 * Performance Benefits:
 * - Direct offset access for common protocols.
 * - No hash lookup overhead for Ethernet/IPv4/UDP/RTP headers.
 */
class MediaPacketParser : public PacketParser
{
public:
    /**
     * @brief: Constructor creates a parser based on the input type.
     *
     * @param [in] include_network_headers: If true, sets up full media streaming chain
     *                                     (Ethernet → IPv4 → UDP → RTP). If false,
     *                                     sets up RTP-only chain. Defaults to true.
     */
    explicit MediaPacketParser(bool include_network_headers = true)
    {
        if (include_network_headers) {
            add_parser(std::make_unique<EthernetParser>());
            add_parser(std::make_unique<IPv4Parser>());
            add_parser(std::make_unique<UDPParser>());
            add_parser(std::make_unique<RTPParser>());
        } else {
            add_parser(std::make_unique<RTPParser>());
        }
    }

    /**
     * @brief: Direct access to Ethernet header.
     *
     * @param [in] data: Packet data buffer.
     * @return: Pointer to Ethernet header, or nullptr if not present.
     */
    const EthernetHeader* ethernet(const byte_t* data) const noexcept
    {
        return m_ethernet_accessor(data);
    }

    /**
     * @brief: Direct access to IPv4 header.
     *
     * @param [in] data: Packet data buffer.
     * @return: Pointer to IPv4 header, or nullptr if not present.
     */
    const IPv4Header* ipv4(const byte_t* data) const noexcept
    {
        return m_ipv4_accessor(data);
    }

    /**
     * @brief: Direct access to UDP header.
     *
     * @param [in] data: Packet data buffer.
     * @return: Pointer to UDP header, or nullptr if not present.
     */
    const UDPHeader* udp(const byte_t* data) const noexcept
    {
        return m_udp_accessor(data);
    }

    /**
     * @brief: Direct access to RTP header.
     *
     * @param [in] data: Packet data buffer.
     * @return: Pointer to RTP header, or nullptr if not present.
     */
    const RTPHeader* rtp(const byte_t* data) const noexcept
    {
        return m_rtp_accessor(data);
    }

    /**
     * @brief: Extract RTP sequence number from packet.
     *
     * This method performs direct sequence number extraction without packet validation
     * for maximum performance. If packet validation is needed, call @ref PacketParser::validate() first.
     *
     * @param [in] data: Packet data buffer.
     * @param [in] length: Packet length.
     * @param [in] is_extended: Whether to extract extended sequence number.
     * @param [out] sequence_number: Extracted sequence number.
     * @return: true if sequence number extraction succeeded.
     */
    bool get_sequence_number(const byte_t* data, size_t length, bool is_extended, uint32_t& sequence_number) const;

    /**
     * @brief: Add a parser to the protocol chain and update cached header accessors.
     *
     * @tparam ParserType: Type of parser to add.
     * @param [in] parser: Unique pointer to the parser.
     */
    template<typename ParserType>
    void add_parser(std::unique_ptr<ParserType> parser)
    {
        PacketParser::add_parser(std::move(parser));
        update_cached_offsets();
    }
private:
    /**
     * @brief: Pre-bound accessor functions for zero-branching performance.
     */
    std::function<const EthernetHeader*(const byte_t*)> m_ethernet_accessor;
    std::function<const IPv4Header*(const byte_t*)> m_ipv4_accessor;
    std::function<const UDPHeader*(const byte_t*)> m_udp_accessor;
    std::function<const RTPHeader*(const byte_t*)> m_rtp_accessor;

    /**
     * @brief: Extract extended sequence number from RTP header.
     *
     * @param [in] rtp_header: Pointer to RTP header.
     * @param [in] data: Full packet data (for bounds checking).
     * @param [in] length: Packet length.
     * @return: Extended sequence number.
     */
    uint16_t get_extended_sequence_number_from_rtp(const RTPHeader* rtp_header, const byte_t* data, size_t length) const;

    /**
     * @brief: Setup pre-bound accessor functions (simplified and optimized).
     *
     * This method creates zero-branching lambda functions that capture offsets.
     */
    void update_cached_offsets()
    {
        auto null_parser = [](const byte_t*) { return nullptr; };

        m_ethernet_accessor = [null_parser](const byte_t* data) -> const EthernetHeader* { 
            return static_cast<const EthernetHeader*>(null_parser(data)); 
        };
        m_ipv4_accessor = [null_parser](const byte_t* data) -> const IPv4Header* { 
            return static_cast<const IPv4Header*>(null_parser(data)); 
        };
        m_udp_accessor = [null_parser](const byte_t* data) -> const UDPHeader* { 
            return static_cast<const UDPHeader*>(null_parser(data)); 
        };
        m_rtp_accessor = [null_parser](const byte_t* data) -> const RTPHeader* { 
            return static_cast<const RTPHeader*>(null_parser(data)); 
        };

        setup_accessor_if_present<EthernetHeader>(m_ethernet_accessor);
        setup_accessor_if_present<IPv4Header>(m_ipv4_accessor);
        setup_accessor_if_present<UDPHeader>(m_udp_accessor);
        setup_accessor_if_present<RTPHeader>(m_rtp_accessor);
    }

    /**
     * @brief: Helper to setup accessor for a specific header type if present.
     */
    template<typename HeaderType>
    void setup_accessor_if_present(std::function<const HeaderType*(const byte_t*)>& accessor)
    {
        size_t parser_index = get_parser_index<HeaderType>();

        if (parser_index != SIZE_MAX) {
            size_t offset = get_parser_offset(parser_index);
            accessor = [offset](const byte_t* data) -> const HeaderType* {
                return reinterpret_cast<const HeaderType*>(data + offset);
            };
        }
    }
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif // RDK_SERVICES_PROTOCOL_MEDIA_PACKET_PARSER_H_
