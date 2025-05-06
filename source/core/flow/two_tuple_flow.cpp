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

#include "rdk/core/flow/flow_interface.h"
#include "rdk/core/flow/two_tuple_flow.h"

using namespace rivermax::dev_kit::core;

TwoTupleFlow::TwoTupleFlow(size_t id, const std::string& ip, uint16_t port) :
    IFlow(id),
    m_ip(ip),
    m_port(port)
{
    IFlow::set_socket_address(m_ip, m_port, m_address);
}
