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

#include <ostream>

#include "rdk/core/stream/send/send_stream_interface.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

ISendStream::ISendStream(const TwoTupleFlow& local_address) :
    ISingleStream(local_address),
    m_num_of_chunks(0)
{
}

std::ostream& ISendStream::print(std::ostream& out) const
{
    return ISingleStream::print(out)
        << "| Number of chunks: " << m_num_of_chunks << "\n";
}
