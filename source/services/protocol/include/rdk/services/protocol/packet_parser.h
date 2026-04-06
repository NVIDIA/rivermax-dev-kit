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

#ifndef RDK_SERVICES_PROTOCOL_PACKET_PARSER_H_
#define RDK_SERVICES_PROTOCOL_PACKET_PARSER_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include <typeindex>
#include <string>
#include <type_traits>
#include <stdexcept>

#include "rdk/services/protocol/protocol_parsers.h"

namespace rdk
{
namespace services
{
/**
 * @brief High-level packet parser for layered protocol stacks.
 *
 * The PacketParser class provides a flexible and efficient mechanism for parsing
 * packets composed of multiple protocol layers. It manages an ordered chain of
 * protocol parsers, each responsible for a specific header, and enables type-safe,
 * zero-copy access to any header within the packet.
 *
 * Key Features:
 * - **Composable Protocol Stack:** Build a custom protocol stack by adding protocol parsers
 *   in the order they appear on the wire.
 * - **Precomputed Header Offsets:** Automatically calculates and caches header offsets for
 *   fast, direct access to any header type.
 * - **Type-Safe Header Access:** Retrieve any header in the chain using its C++ type,
 *   eliminating manual offset calculations and unsafe casts.
 * - **Optional Packet Validation:** Provides packet validation by running all protocol parsers
 *   in sequence, ensuring structural and semantic correctness. Validation can be skipped
 *   for maximum performance in hot data paths.
 *
 * Typical Usage:
 * @code
 *   PacketParser parser;
 *   parser.add_parser(std::make_unique<EthernetParser>());
 *   parser.add_parser(std::make_unique<IPv4Parser>());
 *   parser.add_parser(std::make_unique<UdpParser>());
 *   parser.add_parser(std::make_unique<RtpParser>());
 *
 *   if (parser.validate(packet_data, packet_length)) {
 *       const IPv4Header* ip = parser.get_header<IPv4Header>(packet_data);
 *       // ... process headers ...
 *   }
 * @endcode
 *
 * @note
 * - Parsers must be added in the order headers appear in the packet.
 * - The PacketParser does not own or copy packet data; it provides zero-copy access.
 */
class PacketParser
{
public:
    /**
     * @brief: Default constructor creates an empty chain.
     */
    PacketParser() = default;

    /**
     * @brief: Add a protocol parser to the end of the chain.
     *
     * This is the main method for building your protocol stack. Parsers are added in the order they appear in packets.
     * Each header type can only be added once - attempting to add duplicate types will throw an exception.
     *
     * @tparam ParserType: Type of parser inheriting from @ref IProtocolParser.
     * @param [in] parser: Unique pointer to the parser instance.
     * @throws std::invalid_argument if a parser for this header type already exists in the chain.
     */
    template<typename ParserType>
    void add_parser(std::unique_ptr<ParserType> parser)
    {
        static_assert(std::is_base_of<IProtocolParser, ParserType>::value,
                      "ParserType must inherit from IProtocolParser");
        using HeaderType = typename ParserType::header_type;
        std::type_index header_type_index(typeid(HeaderType));

        if (m_type_to_index.find(header_type_index) != m_type_to_index.end()) {
            throw std::invalid_argument("Parser for this header type is already added to the chain");
        }

        size_t parser_index = m_parsers.size();
        m_type_to_index[header_type_index] = parser_index;

        m_parsers.push_back(std::move(parser));
        recompute_offsets();
        setup_header_accessor<HeaderType>(header_type_index, parser_index);
    }

    /**
     * @brief: Get header by type.
     *
     * @tparam HeaderType: The header struct type.
     * @param [in] data: Packet data buffer.
     * @return: Typed pointer to header, or nullptr if type not found in chain.
     */
    template<typename HeaderType>
    const HeaderType* get_header(const byte_t* data) const
    {
        std::type_index target_type(typeid(HeaderType));
        auto it = m_header_accessors.find(target_type);

        if (it != m_header_accessors.end()) {
            const void* header_ptr = it->second(data);
            return static_cast<const HeaderType*>(header_ptr);
        }

        return nullptr;
    }

    /**
     * @brief: Validate entire packet against the protocol chain.
     *
     * @param [in] data: Packet data buffer.
     * @param [in] length: Packet length in bytes.
     * @return: true if packet is valid for this protocol stack.
     */
    bool validate(const byte_t* data, size_t length) const;

    /**
     * @brief: Check if a header type is supported by this parser.
     *
     * @tparam HeaderType: Header type to check.
     * @return: true if this header type is in the chain.
     */
    template<typename HeaderType>
    bool supports_header_type() const
    {
        std::type_index target_type(typeid(HeaderType));
        return m_type_to_index.find(target_type) != m_type_to_index.end();
    }

    /**
     * @brief: Get the number of parsers in this chain.
     *
     * @return: Number of parsers in the chain.
     */
    size_t get_parser_count() const
    {
        return m_parsers.size();
    }

    /**
     * @brief: Get protocol name by index.
     *
     * @param [in] index: Parser index.
     * @return: Protocol name or empty string if index is invalid.
     */
    std::string get_protocol_name(size_t index) const
    {
        if (index >= m_parsers.size()) {
            return "";
        }
        return m_parsers[index]->get_protocol_name();
    }

    /**
     * @brief: Get minimum packet size for this protocol stack.
     *
     * @return: Minimum size in bytes.
     */
    size_t get_min_packet_size() const
    {
        return m_min_packet_size;
    }
protected:
    /**
     * @brief: Get offset for parser at given index (for specialized subclasses).
     *
     * @param [in] index: Parser index in the chain.
     * @return: Byte offset of the parser's header, or 0 if index is invalid.
     */
    size_t get_parser_offset(size_t index) const
    {
        if (index >= m_offsets.size()) {
            return 0;
        }
        return m_offsets[index];
    }

    /**
     * @brief: Get parser index for a given header type (for specialized subclasses).
     *
     * @tparam HeaderType: The header type to find.
     * @return: Parser index, or SIZE_MAX if not found.
     */
    template<typename HeaderType>
    size_t get_parser_index() const
    {
        std::type_index type_idx(typeid(HeaderType));
        auto it = m_type_to_index.find(type_idx);
        return (it != m_type_to_index.end()) ? it->second : SIZE_MAX;
    }
private:
    /**
     * @brief: The chain of protocol parsers in packet order.
     *
     * Each parser handles exactly one protocol layer. The order matters:
     * parsers[0] handles the first header in the packet,
     * parsers[1] handles the second header, etc.
     */
    std::vector<std::unique_ptr<IProtocolParser>> m_parsers;

    /**
     * @brief: Pre-computed byte offsets for each header.
     */
    std::vector<size_t> m_offsets;

    /**
     * @brief: Pre-bound header accessor functions.
     */
    std::unordered_map<std::type_index, std::function<const void*(const byte_t*)>> m_header_accessors;

    /**
     * @brief: Type registry for header type to parser index mapping.
     */
    std::unordered_map<std::type_index, size_t> m_type_to_index;

    /**
     * @brief: Minimum packet size required for this protocol stack.
     */
    size_t m_min_packet_size = 0;

    /**
     * @brief: Recompute header offsets after adding parsers.
     */
    void recompute_offsets();

    /**
     * @brief: Create pre-bound accessor function for a header type.
     *
     * @tparam HeaderType: The header type.
     * @param [in] type_index: Type index for the header type.
     * @param [in] parser_index: Index of parser in the chain.
     */
    template<typename HeaderType>
    void setup_header_accessor(std::type_index type_index, size_t parser_index)
    {
        size_t offset = m_offsets[parser_index];
        auto accessor_function = [offset](const byte_t* data) -> const void* {
            return data + offset;
        };

        m_header_accessors[type_index] = accessor_function;
    }
};

} // namespace services
} // namespace rdk

#endif // RDK_SERVICES_PROTOCOL_PACKET_PARSER_H_
