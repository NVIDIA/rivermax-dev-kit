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
#include <cstring>

/* Platform specific headers and declarations */
#if defined(_WIN32)
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#elif defined(__linux__)
#include <arpa/inet.h>
#endif

#include "rdk/core/flow/receive_flow.h"

using namespace rivermax::dev_kit::core;


ReceiveFlow::ReceiveFlow(
    size_t id, const std::string& source_ip, uint16_t source_port,
    const std::string& destination_ip, uint16_t destination_port,
    uint32_t tag /*= 0*/) :
    FourTupleFlow(id, source_ip, source_port, destination_ip, destination_port),
    m_tag(tag),
    m_flow_type(ReceiveFlow::Type::FOUR_TUPLE)
{
}

ReceiveFlow::ReceiveFlow(size_t id, const TwoTupleFlow& source_flow, const TwoTupleFlow& destination_flow, uint32_t tag /*= 0*/) :
    FourTupleFlow(id, source_flow, destination_flow),
    m_tag(tag),
    m_flow_type(ReceiveFlow::Type::FOUR_TUPLE)
{
}

ReceiveFlow::ReceiveFlow(size_t id, const std::string& destination_ip, uint16_t destination_port, uint32_t tag /*= 0*/) :
    FourTupleFlow(id, "", 0, destination_ip, destination_port),
    m_tag(tag),
    m_flow_type(ReceiveFlow::Type::TWO_TUPLE)
{
}

ReceiveFlow::ReceiveFlow(size_t id, const TwoTupleFlow& destination_flow, uint32_t tag /*= 0*/) :
    FourTupleFlow(id, TwoTupleFlow(0, "", 0), destination_flow),
    m_tag(tag),
    m_flow_type(ReceiveFlow::Type::TWO_TUPLE)
{
}

const rmx_input_flow& ReceiveFlow::init_descriptor()
{
    rmx_input_init_flow(&m_flow_descriptor);
    rmx_input_set_flow_local_addr(&m_flow_descriptor, &get_destination_socket_address());
    if (m_flow_type == ReceiveFlow::Type::FOUR_TUPLE) {
        rmx_input_set_flow_remote_addr(&m_flow_descriptor, &get_source_socket_address());
    }
    rmx_input_set_flow_tag(&m_flow_descriptor, m_tag);

    m_is_initialized = true;
    return m_flow_descriptor;
}

ReceiveFlow::~ReceiveFlow()
{
    m_is_initialized = false;
}
