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

#ifndef RDK_CORE_FLOW_TWO_TUPLE_FLOW_H_
#define RDK_CORE_FLOW_TWO_TUPLE_FLOW_H_

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

#include "rdk/services/error_handling/return_status.h"
#include "rdk/core/flow/flow_interface.h"

using namespace rivermax::dev_kit::services;

namespace rivermax
{
namespace dev_kit
{
namespace core
{

/**
 * @brief: Two tuple flow interface.
 *
 * This interfaces represents 2 tuple network flow.
 * It can be used to represent send/receive flows.
 */
class TwoTupleFlow : public IFlow
{
template <typename T> friend struct std::hash;
private:
    std::string m_ip;
    uint16_t m_port;
    sockaddr_in m_address;
public:
    /**
     * @brief: TwoTupleFlow constructor.
     *
     * @param [in] id: ID.
     * @param [in] ip: IP address.
     * @param [in] port: Port number.
     */
    TwoTupleFlow(size_t id, const std::string& ip, uint16_t port);
    virtual ~TwoTupleFlow() = default;
    /**
     * @brief: Returns network flow IP.
     *
     * @returns: Network flow IP string.
     */
    const std::string& get_ip() const { return m_ip; }
    /**
     * @brief: Returns network flow port.
     *
     * @returns: Network flow port number.
     */
    uint16_t get_port() const { return m_port; }
    /**
     * @brief: Returns socket address representation of the flow.
     *
     * @returns: Socket address represents the flow.
     */
    sockaddr& get_socket_address() { return reinterpret_cast<sockaddr&>(m_address); }
    const sockaddr& get_socket_address() const { return reinterpret_cast<const sockaddr&>(m_address); }
    /**
     * @brief: Equality operator.
     *
     * @return: true if operands are equal.
     */
    bool operator==(const TwoTupleFlow& rhs) const noexcept
    {
        return std::tie(m_ip, m_port) == std::tie(rhs.m_ip, rhs.m_port);
    }
};

} // namespace core
} // namespace dev_kit
} // namespace rivermax

namespace std
{
/**
 * @brief: Hash specialization for @ref TwoTupleFlow.
 */
template<>
struct hash<rivermax::dev_kit::core::TwoTupleFlow>
{
    /**
     * @brief: Hash function for @ref TwoTupleFlow.
     *
     * @param [in] flow: Reference to @ref TwoTupleFlow object.
     *
     * @return: Hash code.
     */
    std::size_t operator()(const rivermax::dev_kit::core::TwoTupleFlow& flow) const noexcept
    {
        std::size_t h1 = std::hash<std::string>{}(flow.m_ip);
        std::size_t h2 = std::hash<uint16_t>{}(flow.m_port);
        return h1 ^ (h2 << 1);
    }
};

} // namespace std

#endif /* RDK_CORE_FLOW_TWO_TUPLE_FLOW_H_ */
