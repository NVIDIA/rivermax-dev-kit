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

#ifndef RDK_CORE_FLOW_FLOW_INTERFACE_H_
#define RDK_CORE_FLOW_FLOW_INTERFACE_H_

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

using namespace rivermax::dev_kit::services;

namespace rivermax
{
namespace dev_kit
{
namespace core
{

/**
 * @brief: Flow interface.
 *
 * This interfaces represents network flow.
 */
class IFlow
{
private:
    size_t m_id;
public:
    /**
     * @brief: Flow constructor.
     *
     * @param [in] id: Flow ID.
     */
    IFlow(size_t id);
    virtual ~IFlow() = default;
    /**
     * @brief: Returns flow ID.
     *
     * @returns: Flow ID.
     */
    size_t get_id() const { return m_id; }
    /**
     * @brief: Sets socket address.
     *
     * The method sets IP and port to sockaddr_in type address.
     *
     * @param [in] ip: Address IP.
     * @param [in] port: Address port.
     * @param [out] address: Address reference to set.
     *
     * @returns: Status of the operation.
     */
    static ReturnStatus set_socket_address(const std::string& ip, uint16_t port, sockaddr_in& address);
};

} // namespace core
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_CORE_FLOW_FLOW_INTERFACE_H_ */
