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

#include <cstddef>
#include <string>
#include <cstring>
#include <iostream>

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
using namespace rivermax::dev_kit::core;


IFlow::IFlow(size_t id) :
    m_id(id)
{
}

ReturnStatus IFlow::set_socket_address(const std::string& ip, uint16_t port, sockaddr_in& address)
{
    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons(port);

    int rc = inet_pton(AF_INET, ip.c_str(), &address.sin_addr);
    if (rc != 1) {
        std::cerr << "Failed to parse network address" << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}
