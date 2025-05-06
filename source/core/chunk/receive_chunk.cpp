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

#include <cstring>
#include <iostream>

#include <rivermax_api.h>

#include "rdk/core/chunk/receive_chunk.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

ReceiveChunk::ReceiveChunk(rmx_stream_id id, bool use_hds) :
    m_hds_on(use_hds),
    m_data_block_idx(use_hds ? 1 : 0),
    m_stream_id(id)
{
    std::memset(&m_rmax_comp, 0, sizeof(m_rmax_comp));
    rmx_input_init_chunk_handle(&m_chunk_handle, id);
}

ReturnStatus ReceiveChunk::get_next_chunk()
{
    rmx_status status = rmx_input_get_next_chunk(&m_chunk_handle);
    m_rmax_comp = nullptr;
    switch (status) {
    case RMX_OK:
        m_rmax_comp = rmx_input_get_chunk_completion(&m_chunk_handle);
        return ReturnStatus::success;
    case RMX_SIGNAL:
        return ReturnStatus::signal_received;
    default:
        std::cerr << "Failed to get next chunk of stream (" << m_stream_id <<
            "), with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
}
