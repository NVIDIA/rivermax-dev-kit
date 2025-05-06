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

#ifndef RDK_CORE_FLOW_FOUR_TUPLE_FLOW_H_
#define RDK_CORE_FLOW_FOUR_TUPLE_FLOW_H_

#include <cstddef>
#include <string>
#include <tuple>

/* Platform specific headers and declarations */
#if defined(_WIN32)
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#elif defined(__linux__)
#include <arpa/inet.h>
#endif

#include "rdk/core/flow/flow_interface.h"
#include "rdk/core/flow/two_tuple_flow.h"

using namespace rivermax::dev_kit::services;

namespace rivermax
{
namespace dev_kit
{
namespace core
{

/**
 * @brief: Four tuple flow interface.
 *
 * This interfaces represents 4 tuple network flow.
 */
class FourTupleFlow : public IFlow
{
template <typename T> friend struct std::hash;
private:
    TwoTupleFlow m_source_flow;
    TwoTupleFlow m_destination_flow;

public:
    /**
     * @brief: FourTupleFlow constructor.
     *
     * @param [in] id: ID.
     * @param [in] source_ip: Source IP address.
     * @param [in] source_port: Source port number.
     * @param [in] destination_ip: Destination IP address.
     * @param [in] destination_port: Destination port number.
     */
    FourTupleFlow(
        size_t id, const std::string& source_ip, uint16_t source_port,
        const std::string& destination_ip, uint16_t destination_port);
    /**
     * @brief: FourTupleFlow constructor.
     *
     * @param [in] id: ID.
     * @param [in] source_flow: Source flow of the 4 tuple.
     * @param [in] destination_flow: Destination flow of the 4 tuple.
     */
    FourTupleFlow(size_t id, const TwoTupleFlow& source_flow, const TwoTupleFlow& destination_flow);
    virtual ~FourTupleFlow() = default;
    /**
     * @brief: Returns source IP.
     *
     * @returns: Source flow IP string.
     */
    const std::string& get_source_ip() const { return m_source_flow.get_ip(); }
    /**
     * @brief: Returns source port.
     *
     * @returns: Source flow port number.
     */
    uint16_t get_source_port() const { return m_source_flow.get_port(); };
    /**
     * @brief: Returns destination IP.
     *
     * @returns: Destination flow IP string.
     */
    const std::string& get_destination_ip() const { return m_destination_flow.get_ip(); }
    /**
     * @brief: Returns destination port.
     *
     * @returns: Destination flow port number.
     */
    uint16_t get_destination_port() const { return m_destination_flow.get_port(); };
    /**
     * @brief: Returns source flow of the 4 tuple.
     *
     * @returns: Source flow.
     */
    const TwoTupleFlow& get_source_flow() const { return m_source_flow; }
    /**
     * @brief: Returns destination flow of the 4 tuple.
     *
     * @returns: Destination flow.
     */
    const TwoTupleFlow& get_destination_flow() const { return m_destination_flow; }
    /**
     * @brief: Returns socket address representation of the source flow.
     *
     * @returns: Socket address represents source flow.
     */
    sockaddr& get_source_socket_address() { return m_source_flow.get_socket_address(); };
    const sockaddr& get_source_socket_address() const { return m_source_flow.get_socket_address(); };
    /**
     * @brief: Returns socket address representation of the destination flow.
     *
     * @returns: Socket address represents destination flow.
     */
    sockaddr& get_destination_socket_address() { return m_destination_flow.get_socket_address(); };
    const sockaddr& get_destination_socket_address() const { return m_destination_flow.get_socket_address(); };
    /**
     * @brief: Equality operator.
     *
     * @return: true if operands are equal.
     */
    bool operator==(const FourTupleFlow& rhs) const noexcept
    {
        return std::tie(m_source_flow, m_destination_flow) == std::tie(rhs.m_source_flow, rhs.m_destination_flow);
    }
};

} // namespace core
} // namespace dev_kit
} // namespace rivermax

namespace std
{
/**
 * @brief: Hash specialization for @ref FourTupleFlow.
 */
template<>
struct hash<rivermax::dev_kit::core::FourTupleFlow>
{
    /**
     * @brief: Hash function for @ref FourTupleFlow.
     *
     * @param [in] flow: Reference to @ref FourTupleFlow object.
     *
     * @return: Hash code.
     */
    std::size_t operator()(const rivermax::dev_kit::core::FourTupleFlow& flow) const noexcept
    {
        std::size_t h1 = std::hash<rivermax::dev_kit::core::TwoTupleFlow>{}(flow.m_source_flow);
        std::size_t h2 = std::hash<rivermax::dev_kit::core::TwoTupleFlow>{}(flow.m_destination_flow);
        return h1 ^ (h2 << 1);
    }
};
} // namespace std

#endif /* RDK_CORE_FLOW_FOUR_TUPLE_FLOW_H_ */
