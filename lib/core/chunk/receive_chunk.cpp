/*
 * Copyright © 2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
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
