/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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

#include "core/flow/flow.h"

using namespace ral::lib::core;

IFlow::IFlow(size_t id) :
    m_id(id)
{
}

ReturnStatus IFlow::set_socket_address(const std::string& ip, uint16_t port, sockaddr_in& address)
{
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons(port);

    int rc = inet_pton(AF_INET, ip.c_str(), &address.sin_addr);
    if (rc != 1) {
        std::cerr << "Failed to parse network address" << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

TwoTupleFlow::TwoTupleFlow(size_t id, const std::string& ip, uint16_t port) :
    IFlow(id),
    m_ip(ip),
    m_port(port)
{
    IFlow::set_socket_address(m_ip, m_port, m_address);
}

FourTupleFlow::FourTupleFlow(
    size_t id, const std::string& source_ip, uint16_t source_port,
    const std::string& destination_ip, uint16_t destination_port) :
    IFlow(id),
    m_source_flow(TwoTupleFlow(id, source_ip, source_port)),
    m_destination_flow(TwoTupleFlow(id, destination_ip, destination_port))
{
}

FourTupleFlow::FourTupleFlow(
    size_t id, const TwoTupleFlow& source_flow, const TwoTupleFlow& destination_flow) :
    IFlow(id),
    m_source_flow(source_flow),
    m_destination_flow(destination_flow)
{
}
