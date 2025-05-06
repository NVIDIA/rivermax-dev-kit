/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstring>
#include <memory>
#include <iostream>

#include <rivermax_api.h>

#include "core/chunk/receive_chunk.h"

using namespace ral::lib::core;

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
