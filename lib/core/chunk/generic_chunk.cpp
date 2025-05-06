/*
 * Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
#include <memory>
#include <unordered_set>
#include <iostream>

#include <rivermax_api.h>
#include "services/utils/defs.h"
#include "core/chunk/generic_chunk.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

GenericChunk::GenericChunk(rmx_stream_id stream_id, size_t packets_in_chunk) :
    m_stream_id(stream_id)
{
    m_packets.resize(packets_in_chunk, GenericPacket(RMX_MAX_SUB_BLOCKS_PER_MEM_BLOCK));
    rmx_output_gen_init_chunk_handle(&m_chunk, stream_id);
}

ReturnStatus GenericChunk::apply_packets_layout()
{
    for (auto& packet : m_packets) {
        rmx_status status = rmx_output_gen_append_packet_to_chunk(&m_chunk, packet.data(), packet.size());
        if (status != RMX_OK) {
            if (status == RMX_SIGNAL) {
                return ReturnStatus::signal_received;
            }
            
            std::cerr << "Failed to append a packet to a chunk of stream (" << m_stream_id <<
                "), with status: " << status << std::endl;
            return ReturnStatus::failure;
        }
    }

    return ReturnStatus::success;
}

ReturnStatus GenericChunk::get_next_chunk()
{
    rmx_status status = rmx_output_gen_get_next_chunk(&m_chunk);
    switch (status) {
    case RMX_OK:
        return ReturnStatus::success;
    case RMX_SIGNAL:
        return ReturnStatus::signal_received;
    case RMX_NO_FREE_CHUNK:
        return ReturnStatus::no_free_chunks;
    default:
        std::cerr << "Failed to get next chunk of stream (" << m_stream_id <<
            "), with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
}

void GenericChunk::set_commit_options(const std::unordered_set<rmx_output_commit_option>& options)
{
    rmx_output_gen_clear_chunk_all_options(&m_chunk);
    for (auto option : options) {
        rmx_output_gen_set_chunk_option(&m_chunk, option);
    }
}

ReturnStatus GenericChunk::commit_chunk(uint64_t time)
{
    rmx_status status = rmx_output_gen_commit_chunk(&m_chunk, time);
    switch (status) {
    case RMX_OK:
        return ReturnStatus::success;
    case RMX_SIGNAL:
        return ReturnStatus::signal_received;
    case RMX_HW_SEND_QUEUE_IS_FULL:
        return ReturnStatus::hw_send_queue_full;
    default:
        std::cerr << "Failed to commit chunk of stream (" << m_stream_id <<
            "), with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
}

ReturnStatus GenericChunk::mark_for_tracking(uint64_t token)
{
    rmx_status status = rmx_output_gen_mark_chunk_for_tracking(&m_chunk, token);
    switch (status) {
    case RMX_OK:
        return ReturnStatus::success;
    default:
        return ReturnStatus::failure;
    }
}

ReturnStatus GenericChunk::poll_for_completion()
{
    rmx_status status = rmx_output_gen_poll_for_completion(&m_chunk);
    switch (status) {
    case RMX_OK:
        return ReturnStatus::success;
    case RMX_SIGNAL:
        return ReturnStatus::signal_received;
    case RMX_BUSY:
        return ReturnStatus::no_completion;
    default:
        return ReturnStatus::failure;
    }
}

ReturnStatus GenericChunk::get_last_completion_info(uint64_t& timestamp, uint64_t& token)
{
    auto completion = rmx_output_gen_get_last_completion(&m_chunk);
    if (!completion) {
        return ReturnStatus::failure;
    }

    token = rmx_output_get_completion_user_token(completion);
    timestamp = rmx_output_get_completion_timestamp(completion);
    return ReturnStatus::success;
}

