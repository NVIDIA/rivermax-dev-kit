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
#include <memory>
#include <iostream>
#include <algorithm>

#include <rivermax_api.h>

#include "rdk/services/error_handling/return_status.h"
#include "rdk/core/chunk/media_chunk.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

MediaChunk::MediaChunk(rmx_stream_id stream_id, size_t packets_in_chunk, bool use_hds) :
    m_data_ptr(nullptr),
    m_length(packets_in_chunk),
    m_hds_on(use_hds),
    m_stream_id(stream_id)
{
    rmx_output_media_init_chunk_handle(&m_chunk, stream_id);
}

ReturnStatus MediaChunk::get_next_chunk()
{
    rmx_status status = rmx_output_media_get_next_chunk(&m_chunk);
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

void* MediaChunk::get_data_ptr()
{
    return rmx_output_media_get_chunk_strides(&m_chunk, m_hds_on ? 1 : 0);
}

void* MediaChunk::get_app_hdr_ptr()
{
    return m_hds_on ? rmx_output_media_get_chunk_strides(&m_chunk, 0) : nullptr;
}

uint16_t* MediaChunk::get_data_sizes_array()
{
    return rmx_output_media_get_chunk_packet_sizes(&m_chunk, m_hds_on ? 1 : 0);
}

uint16_t* MediaChunk::get_app_hdr_sizes_array()
{
    return m_hds_on ? rmx_output_media_get_chunk_packet_sizes(&m_chunk, 0) : nullptr;
}

void MediaChunk::set_commit_option(rmx_output_commit_option option)
{
    rmx_output_media_set_chunk_option(&m_chunk, option);
}

void MediaChunk::clear_commit_option(rmx_output_commit_option option)
{
    rmx_output_media_set_chunk_option(&m_chunk, option);
}

void MediaChunk::clear_all_commit_options()
{
    rmx_output_media_clear_chunk_all_options(&m_chunk);
}

ReturnStatus MediaChunk::commit_chunk(uint64_t time)
{
    rmx_status status = rmx_output_media_commit_chunk(&m_chunk, time);

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

ReturnStatus MediaChunk::cancel_unsent()
{
    rmx_status status = rmx_output_media_cancel_unsent_chunks(&m_chunk);
    switch (status) {
    case RMX_OK:
        return ReturnStatus::success;
    case RMX_SIGNAL:
        return ReturnStatus::signal_received;
    default:
        return ReturnStatus::failure;
    }
}

ReturnStatus MediaChunk::mark_for_tracking(uint64_t token)
{
    rmx_status status = rmx_output_media_mark_chunk_for_tracking(&m_chunk, token);
    switch (status) {
    case RMX_OK:
        return ReturnStatus::success;
    default:
        return ReturnStatus::failure;
    }
}

ReturnStatus MediaChunk::poll_for_completion()
{
    rmx_status status = rmx_output_media_poll_for_completion(&m_chunk);
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

ReturnStatus MediaChunk::get_last_completion_info(uint64_t& timestamp, uint64_t& token)
{
    auto completion = rmx_output_media_get_last_completion(&m_chunk);
    if (!completion) {
        return ReturnStatus::failure;
    }

    token = rmx_output_get_completion_user_token(completion);
    timestamp = rmx_output_get_completion_timestamp(completion);
    return ReturnStatus::success;
}
