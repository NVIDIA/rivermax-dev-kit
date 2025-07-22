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

#ifndef RDK_SERVICES_PROTOCOL_ETHERNET_PARSER_H_
#define RDK_SERVICES_PROTOCOL_ETHERNET_PARSER_H_

#include "rdk/services/protocol/protocol_parsers.h"
#include "rdk/services/protocol/ethernet.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 * @brief: Ethernet protocol parser.
 *
 * Parses Ethernet headers. No validation is performed as Ethernet headers have a fixed structure.
 */
class EthernetParser : public TypedProtocolParser<EthernetHeader>
{
public:
    size_t get_header_size(const byte_t* data) const override
    {
        NOT_IN_USE(data);
        return sizeof(EthernetHeader);
    }

    bool validate(const byte_t* data, size_t available_length) const override
    {
        NOT_IN_USE(data);
        if (available_length < sizeof(EthernetHeader)) {
            return false;
        }

        return true;
    }

    std::string get_protocol_name() const override
    {
        return "Ethernet";
    }
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif // RDK_SERVICES_PROTOCOL_ETHERNET_PARSER_H_
