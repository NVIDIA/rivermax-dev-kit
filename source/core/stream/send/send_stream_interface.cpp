/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <ostream>

#include "rdk/core/stream/send/send_stream_interface.h"

using namespace rdk::services;
using namespace rdk::core;

ISendStream::ISendStream(const std::vector<TwoTupleFlow>& local_addresses) :
    IStream(),
    m_local_addresses(local_addresses)
{
}

std::ostream& ISendStream::print(std::ostream& out) const
{
    out << "| Stream ID: ";
    if (m_stream_created) {
        out << m_stream_id << "\n";
    } else {
        out << "not set\n";
    }
    if (m_local_addresses.size() == 1) {
        out << "| NIC IP: " << m_local_addresses[0].get_ip() << "\n";
    } else {
        out << "| NIC IP(s):\n";
        for (size_t i = 0; i < m_local_addresses.size(); ++i) {
            out << "|   [" << i << "] " << m_local_addresses[i].get_ip() << "\n";
        }
    }

    return out;
}
