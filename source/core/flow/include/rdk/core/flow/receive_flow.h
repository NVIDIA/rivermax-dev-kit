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

#ifndef RDK_CORE_FLOW_RECEIVE_FLOW_H_
#define RDK_CORE_FLOW_RECEIVE_FLOW_H_

#include "rivermax_defs.h"
#include <cstddef>
#include <string>

/* Platform specific headers and declarations */
#if defined(_WIN32)
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#elif defined(__linux__)
#include <arpa/inet.h>
#endif

#include <rivermax_api.h>

#include "rdk/core/flow/four_tuple_flow.h"

namespace rivermax
{
namespace dev_kit
{
namespace core
{
/**
 * @brief: Represents receive flow.
 *
 * This class is used to represent Rivermax receive flow.
 */
class ReceiveFlow : public FourTupleFlow
{
public:
    /**
     * @brief: Receive flow type.
     *
     * This enum is used to define the type of receive flow.
     */
    enum class Type
    {
        /* Represents flow with destination IP and port */
        TWO_TUPLE,
        /* Represents flow with source and destination IP and ports */
        FOUR_TUPLE,
        /* Represents undefined flow type */
        UNDEFINED
    };
private:
    uint32_t m_tag = 0;
    rmx_input_flow m_flow_descriptor = {0};
    bool m_is_initialized = false;
    Type m_flow_type = Type::UNDEFINED;

public:
    /**
     * @brief: Four tuple receive flow constructor.
     *
     * This constructor is used to create a receive flow with 4 tuple,
     * which includes source and destination IP addresses and ports.
     *
     * @param [in] id: ID.
     * @param [in] source_ip: Source IP address.
     * @param [in] source_port: Source port number.
     * @param [in] destination_ip: Destination IP address.
     * @param [in] destination_port: Destination port number.
     * @param [in] tag: Flow tag (optional, default is 0).
     */
    ReceiveFlow(
        size_t id, const std::string& source_ip, uint16_t source_port,
        const std::string& destination_ip, uint16_t destination_port,
        uint32_t tag = 0);
    /**
     * @brief: Four tuple receive flow constructor.
     *
     * This constructor is used to create a receive flow with 4 tuple,
     * which includes source and destination IP addresses and ports.
     *
     * @param [in] id: ID.
     * @param [in] source_flow: Source flow of the 4 tuple.
     * @param [in] destination_flow: Destination flow of the 4 tuple.
     * @param [in] tag: Flow tag (optional, default is 0).
     */
    ReceiveFlow(size_t id, const TwoTupleFlow& source_flow,
        const TwoTupleFlow& destination_flow, uint32_t tag = 0);
    /**
     * @brief: Two tuple receive flow constructor.
     *
     * This constructor is used to create a receive flow with 4 tuple,
     * which includes source and destination IP addresses and ports.
     *
     * @param [in] id: ID.
     * @param [in] destination_ip: Destination IP address.
     * @param [in] destination_port: Destination port number.
     * @param [in] tag: Flow tag (optional, default is 0).
     */
    ReceiveFlow(size_t id, const std::string& destination_ip,
        uint16_t destination_port, uint32_t tag = 0);
    /**
     * @brief: Two tuple receive flow constructor.
     *
     * This constructor is used to create a receive flow with 2 tuple,
     * which includes only destination IP address and port.
     *
     * @param [in] id: ID.
     * @param [in] destination_flow: Destination flow of the 4 tuple.
     * @param [in] tag: Flow tag (optional, default is 0).
     */
    ReceiveFlow(size_t id, const TwoTupleFlow& destination_flow, uint32_t tag = 0);
    ~ReceiveFlow();
    /**
     * @brief: Returns flow tag.
     *
     * @returns: Flow tag.
     */
     uint32_t get_tag() const { return m_tag; }
    /**
     * @brief: Returns flow descriptor.
     *
     * @return: Rivermax input flow descriptor.
     */
    const rmx_input_flow& get_descriptor() const { return m_flow_descriptor; }
    /**
     * @brief: Initialize flow descriptor.
     *
     * @return: Rivermax input flow descriptor.
     */
    const rmx_input_flow& init_descriptor();
    /**
     * @brief: Returns flow type.
     *
     * @return: Flow type.
     */
    ReceiveFlow::Type get_flow_type() const { return m_flow_type; }
};

} // namespace core
} // namespace dev_kit
} // namespace rivermax

namespace std
{
/**
 * @brief: Hash specialization for @ref ReceiveFlow.
 */
template<>
struct hash<rivermax::dev_kit::core::ReceiveFlow>
{
    /**
     * @brief: Hash function for @ref ReceiveFlow.
     *
     * @param [in] flow: Reference to @ref ReceiveFlow object.
     *
     * @return: Hash code.
     */
    std::size_t operator()(const rivermax::dev_kit::core::ReceiveFlow& flow) const noexcept
    {
        return std::hash<rivermax::dev_kit::core::FourTupleFlow>{}(flow) ^
               std::hash<uint32_t>{}(flow.get_tag());
    }
};

} // namespace std

#endif /* RDK_CORE_FLOW_RECEIVE_FLOW_H_ */
