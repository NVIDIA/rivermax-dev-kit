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
#include <chrono>
#include <thread>

#include "rdk/services/utils/defs.h"
#include "rdk/services/media/media_defs.h"
#include "rdk/core/stream/send/media_stream.h"
#include "rdk/core/chunk/media_chunk.h"
#include "rdk/core/flow/flow.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

MediaStreamSettings::MediaStreamSettings(const TwoTupleFlow& local_address,
            const MediaSettings& media_settings,
            size_t packets_per_chunk, uint16_t packet_payload_size,
            size_t data_stride_size, size_t app_header_stride_size,
            uint8_t dscp, uint8_t pcp, uint8_t ecn) :
        IStreamSettings(s_build_steps),
        m_local_address(local_address),
        m_media_settings(media_settings),
        m_packets_per_chunk(packets_per_chunk),
        m_packet_payload_size(packet_payload_size),
        m_data_stride_size(data_stride_size),
        m_app_header_stride_size(app_header_stride_size),
        m_dscp(dscp),
        m_pcp(pcp),
        m_ecn(ecn)
{
}

IStreamSettings<MediaStreamSettings, rmx_output_media_stream_params>::SetterSequence MediaStreamSettings::s_build_steps{
    &MediaStreamSettings::stream_param_init,
    &MediaStreamSettings::stream_param_set_sdp,
    &MediaStreamSettings::stream_param_set_packets_per_chunk,
    &MediaStreamSettings::stream_param_set_stride_sizes,
    &MediaStreamSettings::stream_param_set_packets_per_frame,
    &MediaStreamSettings::stream_param_set_pcp,
    &MediaStreamSettings::stream_param_set_dscp,
    &MediaStreamSettings::stream_param_set_ecn,
};

void MediaStreamSettings::stream_param_init(rmx_output_media_stream_params& descr)
{
    rmx_output_media_init(&descr);
}

void MediaStreamSettings::stream_param_set_sdp(rmx_output_media_stream_params& descr)
{
    rmx_output_media_set_sdp(&descr, m_media_settings.sdp.c_str());
    rmx_output_media_set_idx_in_sdp(&descr, m_media_settings.media_block_index);
}

void MediaStreamSettings::stream_param_set_packets_per_chunk(rmx_output_media_stream_params& descr)
{
    rmx_output_media_set_packets_per_chunk(&descr, m_packets_per_chunk);
}

void MediaStreamSettings::stream_param_set_stride_sizes(rmx_output_media_stream_params& descr)
{
    if (m_app_header_stride_size) {
        rmx_output_media_set_stride_size(&descr, 0, m_app_header_stride_size);
    }
    rmx_output_media_set_stride_size(&descr, m_app_header_stride_size ? 1 : 0, m_data_stride_size);
}

void MediaStreamSettings::stream_param_set_packets_per_frame(rmx_output_media_stream_params& descr)
{
    rmx_output_media_set_packets_per_frame(&descr, m_media_settings.packets_in_frame_field);
}

void MediaStreamSettings::stream_param_set_pcp(rmx_output_media_stream_params& descr)
{
    rmx_output_media_set_pcp(&descr, m_pcp);
}

void MediaStreamSettings::stream_param_set_dscp(rmx_output_media_stream_params& descr)
{
    rmx_output_media_set_dscp(&descr, m_dscp);
}

void MediaStreamSettings::stream_param_set_ecn(rmx_output_media_stream_params& descr)
{
    rmx_output_media_set_ecn(&descr, m_ecn);
}

MediaStreamMemBlockset::MediaStreamMemBlockset(size_t block_count, size_t sub_block_count, size_t chunks_per_block) :
    m_blocks(block_count),
    m_sub_block_count(sub_block_count),
    m_chunks_per_block(chunks_per_block)
{
    rmx_output_media_init_mem_blocks(m_blocks.data(), m_blocks.size());
    for (auto& block : m_blocks) {
        rmx_output_media_set_sub_block_count(&block, sub_block_count);
        rmx_output_media_set_chunk_count(&block, chunks_per_block);
    }
}

void MediaStreamMemBlockset::set_block_memory(size_t idx, size_t sub_block_idx, void* block_memory_start,
            size_t block_memory_size, rmx_mkey_id memory_key)
{
    auto& block = m_blocks[idx];
    rmx_mem_region* region = rmx_output_media_get_sub_block(&block, sub_block_idx);
    region->addr = block_memory_start;
    region->length = block_memory_size;
    region->mkey = memory_key;
}

void MediaStreamMemBlockset::set_rivermax_to_allocate_memory()
{
    for (size_t block_idx = 0; block_idx < m_blocks.size(); block_idx++) {
        for (size_t sub_block_idx = 0; sub_block_idx < m_sub_block_count; sub_block_idx++) {
            set_block_memory(block_idx, sub_block_idx, nullptr, 0, RMX_MKEY_INVALID);
        }
    }
}

void MediaStreamMemBlockset::set_dup_block_memory(size_t idx, size_t sub_block_idx, void* block_memory_start,
            size_t block_memory_size, rmx_mkey_id memory_keys[])
{
    auto& block = m_blocks[idx];
    rmx_mem_multi_key_region* multiregion = rmx_output_media_get_dup_sub_block(&block, sub_block_idx);
    multiregion->addr = block_memory_start;
    multiregion->length = block_memory_size;
    for (size_t i = 0; i < RMX_MAX_DUP_STREAMS; i++) {
        multiregion->mkey[i] = memory_keys[i];
    }
}

ReturnStatus MediaStreamMemBlockset::set_block_layout(size_t idx, uint16_t data_sizes[], uint16_t app_header_sizes[])
{
    if ((m_sub_block_count < 2) && app_header_sizes) {
        std::cerr << "Error setting header sizes when header-data split is not configured." << std::endl;
        return ReturnStatus::failure;
    }
    auto& block = m_blocks[idx];
    if (m_sub_block_count < 2) {
        rmx_output_media_set_packet_layout(&block, 0, data_sizes);
    } else {
        rmx_output_media_set_packet_layout(&block, 0, app_header_sizes);
        rmx_output_media_set_packet_layout(&block, 1, data_sizes);
    }
    return ReturnStatus::success;
}

MediaSendStream::MediaSendStream(const MediaStreamSettings& settings) :
    ISendStream(settings.m_local_address),
    m_stream_settings(settings)
{
    m_stream_settings.build(m_stream_settings, m_stream_params);
    m_num_of_chunks = 0;
}

MediaSendStream::MediaSendStream(const MediaStreamSettings& settings, MediaStreamMemBlockset& mem_blocks) :
    ISendStream(settings.m_local_address),
    m_stream_settings(settings)
{
    m_stream_settings.build(m_stream_settings, m_stream_params);
    assign_memory_blocks(mem_blocks);
}

ReturnStatus MediaSendStream::initialize_memory_layout()
{
    return ReturnStatus::success;
}

ReturnStatus MediaSendStream::determine_memory_layout(
    HeaderPayloadMemoryLayoutRequest& memory_layout_request) const
{
    return ReturnStatus::success;
}

ReturnStatus MediaSendStream::validate_memory_layout(
    const MediaMemoryLayoutResponse& memory_layout_response) const
{
    return ReturnStatus::success;
}

ReturnStatus MediaSendStream::apply_memory_layout(const MediaMemoryLayoutResponse& memory_layout_response)
{
    ReturnStatus status = validate_memory_layout(memory_layout_response);
    if (status != ReturnStatus::success) {
        std::cerr << "Failed to validate memory layout" << std::endl;
        return status;
    }
    assign_memory_blocks(memory_layout_response.mem_blocks);
    return ReturnStatus::success;
}

void MediaSendStream::assign_memory_blocks(MediaStreamMemBlockset& mem_blocks)
{
    m_num_of_chunks = mem_blocks.get_memory_block_count() * mem_blocks.get_chunks_per_block();
    rmx_output_media_assign_mem_blocks(&m_stream_params, mem_blocks.get_memory_blocks(),
            mem_blocks.get_memory_block_count());
}

std::ostream& MediaSendStream::print(std::ostream& out) const
{
    ISendStream::print(out);

    out << "| SDP file: " << "\n"
        << "---------------------------------------------------------------------------------------" << "\n"
        << m_stream_settings.m_media_settings.sdp << "\n"
        << "---------------------------------------------------------------------------------------" << "\n"
        << "+**********************************************\n";

    return out;
}

ReturnStatus MediaSendStream::create_stream()
{
    rmx_status status = rmx_output_media_create_stream(&m_stream_params, &m_stream_id);

    if (status != RMX_OK) {
        std::cerr << "Failed to create media stream with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
    m_stream_created = true;
    return ReturnStatus::success;
}

ReturnStatus MediaSendStream::destroy_stream()
{
    if (!m_stream_created) {
        std::cerr << "Failed to destroy media stream, the stream was not created" << std::endl;
        return ReturnStatus::failure;
    }

    rmx_status status;

    do {
        status = rmx_output_media_destroy_stream(m_stream_id);
        if (status == RMX_BUSY) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } while (status == RMX_BUSY);

    if (status != RMX_OK) {
        std::cerr << "Failed to destroy media stream with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
    m_stream_created = false;
    return ReturnStatus::success;
}

ReturnStatus MediaSendStream::get_next_chunk(MediaChunk& chunk)
{
    return chunk.get_next_chunk();
}

ReturnStatus MediaSendStream::blocking_get_next_chunk(MediaChunk& chunk, size_t retries)
{
    ReturnStatus status;

    do {
        status = get_next_chunk(chunk);
    } while (unlikely(status == ReturnStatus::no_free_chunks && retries--));

    return status;
}

ReturnStatus MediaSendStream::commit_chunk(MediaChunk& chunk, uint64_t time)
{
    return chunk.commit_chunk(time);
}

ReturnStatus MediaSendStream::blocking_commit_chunk(MediaChunk& chunk, uint64_t time, size_t retries)
{
    ReturnStatus status;

    do {
        status = commit_chunk(chunk, time);
    } while (unlikely(status == ReturnStatus::hw_send_queue_full && retries--));

    return status;
}

ReturnStatus MediaSendStream::cancel_unsent_chunks(MediaChunk& chunk)
{
    return chunk.cancel_unsent();
}
