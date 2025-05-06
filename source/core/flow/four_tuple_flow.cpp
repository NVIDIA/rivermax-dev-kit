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
#include "rdk/core/flow/four_tuple_flow.h"

using namespace rivermax::dev_kit::core;

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
