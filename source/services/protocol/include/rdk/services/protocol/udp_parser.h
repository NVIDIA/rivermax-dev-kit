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

#ifndef RDK_SERVICES_PROTOCOL_UDP_PARSER_H_
#define RDK_SERVICES_PROTOCOL_UDP_PARSER_H_

/* Platform specific headers and declarations */
#if defined(_WIN32)
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#elif defined(__linux__)
#include <arpa/inet.h>
#endif

#include "rdk/services/protocol/protocol_parsers.h"
#include "rdk/services/protocol/udp.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 * @brief: UDP protocol parser.
 *
 * Parses UDP headers (always 8 bytes). Validates header length field
 * against available data and checks for minimum valid length.
 */
class UDPParser : public TypedProtocolParser<UDPHeader>
{
public:
    size_t get_header_size(const byte_t* data) const override
    {
        NOT_IN_USE(data);
        return UDP_HEADER_SIZE;
    }

    bool validate(const byte_t* data, size_t available_length) const override
    {
        if (available_length < UDP_HEADER_SIZE) {
            return false;
        }

        const UDPHeader* udp_header = get_typed_header(data);
        uint16_t udp_length = ntohs(udp_header->length);

        if (udp_length < UDP_HEADER_SIZE) {
            return false;
        }

        if (udp_length > available_length) {
            return false;
        }
        return true;
    }

    std::string get_protocol_name() const override
    {
        return "UDP";
    }
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif // RDK_SERVICES_PROTOCOL_UDP_PARSER_H_
