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

#ifndef RDK_SERVICES_PROTOCOL_PROTOCOL_PARSERS_H_
#define RDK_SERVICES_PROTOCOL_PROTOCOL_PARSERS_H_

#include <cstddef>
#include <string>
#include <typeindex>

#include "rdk/services/utils/defs.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 * @brief: Base interface for protocol parsers.
 *
 * This interface enables efficient parsing of network protocol headers without
 * copying data. Parsers validate protocol headers in-place and provide direct
 * access to header fields through typed pointers, allowing high-performance
 * packet processing in networking applications.
 *
 * Key design goals:
 * - Zero-copy parsing: Direct access to packet data without memcpy
 * - Validation (optional): Ensure headers are well-formed before access
 * - Composability: Chain multiple parsers for layered protocols
 * - Type safety: Strongly typed header access through template specialization
 * - Performance: Minimal overhead for high-throughput packet processing
 */
class IProtocolParser
{
public:
    virtual ~IProtocolParser() = default;

    /**
     * @brief: Get the size of this protocol's header.
     *
     * @param [in] data: Pointer to the start of this protocol's header.
     * @return: Size in bytes of the header.
     */
    virtual size_t get_header_size(const byte_t* data) const = 0;

    /**
     * @brief: Validate that the data contains a valid header for this protocol.
     *
     * @param [in] data: Pointer to the start of this protocol's header.
     * @param [in] available_length: How many bytes are available from this position.
     * @return: true if the header appears valid for this protocol.
     */
    virtual bool validate(const byte_t* data, size_t available_length) const = 0;

    /**
     * @brief: Get a pointer to the parsed header structure.
     *
     * @param [in] data: Pointer to the start of this protocol's header.
     * @return: Untyped pointer to the header structure.
     */
    virtual const void* get_header(const byte_t* data) const = 0;

    /**
     * @brief: Get the name of this protocol for debugging and logging.
     *
     * @return: Human-readable protocol name.
     */
    virtual std::string get_protocol_name() const = 0;
protected:
    /**
     * @brief: Protected constructor.
     */
    IProtocolParser() = default;
};

/**
 * @brief: Template base class that provides type-safe header access.
 *
 * This template extends the base interface with compile-time type safety,
 * allowing parsers to work with strongly-typed header structures while
 * maintaining interface compatibility through type erasure.
 *
 * Example usage:
 * @code
 * class EthernetParser : public TypedProtocolParser<EthernetHeader> {
 *   // Implementation automatically gets type-safe get_typed_header() method
 * };
 * @endcode
 *
 * @tparam HeaderType: The specific protocol header structure this parser handles.
 */
template<typename HeaderType>
class TypedProtocolParser : public IProtocolParser
{
public:
    /**
     * @brief: Type alias for the header type this parser handles.
     *
     * This allows the chain manager to know what type this parser works with.
     */
    using header_type = HeaderType;

    /**
     * @brief: Type-safe header access method.
     *
     * @param [in] data: Pointer to the start of this protocol's header.
     * @return: Typed pointer to the header structure.
     */
    const HeaderType* get_typed_header(const byte_t* data) const
    {
        return reinterpret_cast<const HeaderType*>(data);
    }

    /**
     * @brief: Get type information for this header type.
     *
     * @return: std::type_index for HeaderType.
     */
    std::type_index get_header_type_index() const
    {
        return std::type_index(typeid(HeaderType));
    }

    const void* get_header(const byte_t* data) const override
    {
        return get_typed_header(data);
    }
protected:
    TypedProtocolParser() = default;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif // RDK_SERVICES_PROTOCOL_PROTOCOL_PARSERS_H_
