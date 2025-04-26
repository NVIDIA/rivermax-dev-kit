/*
 * Copyright © 2017-2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#include <algorithm>
#include <cstring>
#include <iostream>

#include "core/stream/send/media_stream.h"
#include "core/chunk/media_chunk.h"
#include "core/flow/flow.h"
#include "services/error_handling/error_handling.h"
#include "services/utils/defs.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

MediaSendStream::MediaSendStream(
    const TwoTupleFlow& local_address, const media_settings_t& media_settings,
    rmax_buffer_attr* buffer_attributes, rmax_qos_attr* qos_attributes) :
    ISendStream(local_address),
    m_media_settings(media_settings)
{
    initialize_rmax_stream_parameters(buffer_attributes, qos_attributes);
}

void MediaSendStream::initialize_rmax_stream_parameters(
    rmax_buffer_attr* buffer_attributes, rmax_qos_attr* qos_attributes)
{
    memset(&m_rmax_parameters, 0, sizeof(m_rmax_parameters));

    m_rmax_parameters.flags = 0;
    m_rmax_parameters.sdp_chr = const_cast<char*>(m_media_settings.sdp.c_str());
    m_rmax_parameters.buffer_attr = buffer_attributes;
    m_rmax_parameters.qos = qos_attributes;
    m_rmax_parameters.num_packets_per_frame = m_media_settings.packets_in_frame_field;
    m_rmax_parameters.media_block_index = m_media_settings.media_block_index;
    // TODO: Add support for setting source port.
    m_rmax_parameters.source_port_arr = nullptr;
    m_rmax_parameters.source_port_arr_sz = 0;

    for (size_t index = 0; index < buffer_attributes->mem_block_array_len; index++) {
        m_num_of_chunks += buffer_attributes->mem_block_array[index].chunks_num;
    }
}

std::ostream& MediaSendStream::print(std::ostream& out) const
{
    ISendStream::print(out);

    out << "| SDP file: " << "\n"
        << "---------------------------------------------------------------------------------------" << "\n"
        << m_media_settings.sdp << "\n"
        << "---------------------------------------------------------------------------------------" << "\n"
        << "+**********************************************\n";

    return out;
}

ReturnStatus MediaSendStream::create_stream()
{
    rmax_status_t status = rmax_out_create_stream_ex(&m_rmax_parameters, &m_stream_id);

    if (status != RMAX_OK) {
        std::cerr << "Failed to create media stream with status: " << status << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus MediaSendStream::destroy_stream()
{
    rmax_status_t status = rmax_out_cancel_unsent_chunks(m_stream_id);

    if (status != RMAX_OK) {
        std::cerr << "Failed to cancel media stream with status: " << status << std::endl;
        return ReturnStatus::failure;
    }

    do {
        status = rmax_out_destroy_stream(m_stream_id);
    } while (status == RMAX_ERR_BUSY);

    if (status != RMAX_OK) {
        std::cerr << "Failed to destroy media stream with status: " << status << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus MediaSendStream::get_next_chunk(MediaChunk& chunk)
{
    rmax_status_t status = rmax_out_get_next_chunk(m_stream_id, chunk.get_data_ptr(), chunk.get_app_hdr_ptr());

    switch (status) {
    case RMAX_OK:
        chunk.set_length(m_rmax_parameters.buffer_attr->chunk_size_in_strides);
        return ReturnStatus::success;
    case RMAX_SIGNAL:
        return ReturnStatus::signal_received;
    case RMAX_ERR_NO_FREE_CHUNK:
        return ReturnStatus::no_free_chunks;
    default:
        std::cerr << "Failed to get next chunk of stream (" << m_stream_id <<
            "), with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
}

ReturnStatus MediaSendStream::blocking_get_next_chunk(MediaChunk& chunk, size_t retries)
{
    ReturnStatus status;
    
    do {
        status = get_next_chunk(chunk);
    } while (unlikely(!retries-- || status == ReturnStatus::no_free_chunks));

    return status;
}

ReturnStatus MediaSendStream::commit_chunk(uint64_t timestamp_ns, rmax_commit_flags_t flags) const
{
    rmax_status_t status = rmax_out_commit(m_stream_id, timestamp_ns, flags);

    switch (status) {
    case RMAX_OK:
        return ReturnStatus::success;
    case RMAX_SIGNAL:
        return ReturnStatus::signal_received;
    case RMAX_ERR_HW_SEND_QUEUE_FULL:
        return ReturnStatus::hw_send_queue_full;
    default:
        std::cerr << "Failed to commit chunk of stream (" << m_stream_id <<
            "), with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
}

ReturnStatus MediaSendStream::blocking_commit_chunk(
    uint64_t timestamp_ns, rmax_commit_flags_t flags, size_t retries) const
{
    ReturnStatus rc;

    do {
        rc = commit_chunk(timestamp_ns, flags);
    } while (unlikely(!retries-- || rc == ReturnStatus::hw_send_queue_full));

    return rc;
}
