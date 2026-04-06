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

#ifndef RDK_SERVICES_PROTOCOL_IPV4_PARSER_H_
#define RDK_SERVICES_PROTOCOL_IPV4_PARSER_H_

/* Platform specific headers and declarations */
#if defined(_WIN32)
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#elif defined(__linux__)
#include <arpa/inet.h>
#endif

#include "rdk/services/protocol/protocol_parsers.h"
#include "rdk/services/protocol/ipv4.h"

namespace rdk
{
namespace services
{
/**
 * @brief: IPv4 protocol parser.
 *
 * Parses IPv4 headers with variable size (20-60 bytes) based on the Internet Header Length (IHL) field.
 * Validates version and header length.
 */
class IPv4Parser : public TypedProtocolParser<IPv4Header>
{
public:
    size_t get_header_size(const byte_t* data) const override
    {
        if (!data) {
            return IPV4_MIN_HEADER_SIZE;
        }

        const IPv4Header* ipv4_header = get_typed_header(data);
        uint8_t ihl = ipv4_header->version_ihl & 0x0F;
        size_t header_size = ihl * IPV4_IHL_MULTIPLIER;

        if (header_size < IPV4_MIN_HEADER_SIZE || header_size > IPV4_MAX_HEADER_SIZE) {
            return IPV4_MIN_HEADER_SIZE;
        }

        return header_size;
    }

    bool validate(const byte_t* data, size_t available_length) const override
    {
        if (available_length < IPV4_MIN_HEADER_SIZE) {
            return false;
        }

        const IPv4Header* ipv4_header = get_typed_header(data);

        uint8_t version = (ipv4_header->version_ihl >> 4) & 0x0F;
        if (version != IPV4_VERSION) {
            return false;
        }

        uint8_t ihl = ipv4_header->version_ihl & 0x0F;
        if (ihl < IPV4_MIN_IHL || ihl > IPV4_MAX_IHL) {
            return false;
        }

        size_t actual_header_size = ihl * IPV4_IHL_MULTIPLIER;
        if (available_length < actual_header_size) {
            return false;
        }

        uint16_t total_length = ntohs(ipv4_header->total_length);
        if (total_length < actual_header_size) {
            return false;
        }
        return true;
    }

    std::string get_protocol_name() const override
    {
        return "IPv4";
    }
};

} // namespace services
} // namespace rdk

#endif // RDK_SERVICES_PROTOCOL_IPV4_PARSER_H_
