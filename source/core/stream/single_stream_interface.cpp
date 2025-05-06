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

#include "rdk/core/stream/single_stream_interface.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

ISingleStream::ISingleStream(const TwoTupleFlow& local_address) :
    IStream(),
    m_local_address(local_address)
{
}

std::ostream& ISingleStream::print(std::ostream& out) const
{
    out << "| Stream ID: ";
    if (m_stream_created) {
        out << m_stream_id << "\n";
    } else {
        out << "not set\n";
    }
    out << "| NIC IP: " << m_local_address.get_ip() << "\n";
        // TODO: Enable this after adding set and/or query source port support.
        //<< "| Source port: " << m_local_address.get_source_port() << "\n";

    return out;
}
