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

#include <thread>
#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>
#include <cstring>

#include <rivermax_api.h>
#include "rt_threads.h"

#include "rdk/io_node/senders/media_sender_io_node.h"
#include "rdk/services/error_handling/error_handling.h"
#include "rdk/services/cpu/affinity/affinity.h"

using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

static void replace_all(
    std::string& source_str, const std::string& outer_prefix_str, const std::string& inner_prefix_str,
    const std::string& new_str, const char* suffix_str, std::string::size_type start_replacement_location = 0)
{
    /*
    * Starting at start_replacement_location, repeatedly search for outer_prefix_str then inner_prefix_str,
    * then substitute up to suffix_str with new_str.
    * so e.g., a pattern like <prefix><something-to-keep><infix><anything><suffix>
    * can be transformed into <prefix><something-to-keep><infix><replaced><suffix>
    */
    std::string::size_type n = start_replacement_location;
    std::string::size_type n2 = 0;
    while ((n = source_str.find(outer_prefix_str, n)) != std::string::npos) {
        n = source_str.find(inner_prefix_str, n + outer_prefix_str.length());
        if (n == std::string::npos) {
            break;
        }
        n2 = source_str.find(suffix_str, n + inner_prefix_str.length());
        if (n2 == std::string::npos) {
            break;
        }
        source_str.replace(n + inner_prefix_str.length(), n2 - (n + inner_prefix_str.length()), new_str);
        n += inner_prefix_str.length();
    }
}

/**
* @breif: Replace all occurrences of sub string in the input string.
*
* @pram [in] source_str: Source string.
* @pram [in] prefix_str: Prefix string before source_str.
* @pram [in] new_str: The new string to replace.
* @pram [in] suffix_str: Suffix string after source_str.
* @pram [in] start_replacement_location: The location in the string to start the replacement from, defaults to 0.
*/
static inline void replace_all(
    std::string& source_str, const std::string& prefix_str,
    const std::string& new_str, const char* suffix_str,
    std::string::size_type start_replacement_location = 0)
{
    return replace_all(source_str, prefix_str, "", new_str, suffix_str, start_replacement_location);
}

MediaSenderIONode::MediaSenderIONode(
        const FourTupleFlow& network_address,
        std::shared_ptr<AppSettings> app_settings,
        size_t index, size_t num_of_streams, int cpu_core_affinity,
        IONodeMemoryUtils& memory_utils,
        time_handler_ns_cb_t time_hanlder_cb) :
    m_stream_packs(num_of_streams),
    m_media_settings(app_settings->media),
    m_video_file(app_settings->video_file),
    m_index(index),
    m_network_address(network_address),
    m_sleep_between_operations(app_settings->sleep_between_operations),
    m_print_parameters(app_settings->print_parameters),
    m_cpu_core_affinity(cpu_core_affinity),
    m_hw_queue_full_sleep_us(app_settings->hw_queue_full_sleep_us),
    m_memory_utils(memory_utils),
    m_num_of_memory_blocks(app_settings->num_of_memory_blocks),
    m_num_of_chunks_in_mem_block(app_settings->num_of_chunks_in_mem_block),
    m_packet_header_size(app_settings->packet_app_header_size),
    m_packet_payload_size(app_settings->packet_payload_size),
    m_num_of_packets_in_chunk(app_settings->num_of_packets_in_chunk),
    m_num_of_packets_in_mem_block(app_settings->num_of_packets_in_mem_block),
    m_app_header_stride_size(align_up_pow2(m_packet_header_size, get_cache_line_size())),
    m_data_stride_size(align_up_pow2(m_packet_payload_size, get_cache_line_size())),
    m_header_total_memory_size(0),
    m_payload_total_memory_size(0),
    m_block_payload_memory_size(0),
    m_block_header_memory_size(0),
    m_dscp(0), m_pcp(0), m_ecn(0),
    m_get_time_ns_cb(std::move(time_hanlder_cb)),
    m_gpu_enabled(app_settings->gpu_id != INVALID_GPU_ID),
    m_dynamic_video_file_load(app_settings->dynamic_video_file_load)
{
    m_stream_packs.resize(num_of_streams);
    m_num_of_memory_blocks = DEFAULT_NUMBER_OF_MEM_BLOCKS;
    initialize_memory_layout();
}

std::ostream& MediaSenderIONode::print(std::ostream& out) const
{
    out << "+#############################################\n"
        << "| Sender index: " << m_index << "\n"
        << "| Thread ID: 0x" << std::hex << std::this_thread::get_id() << std::dec << "\n"
        << "| CPU core affinity: " << m_cpu_core_affinity << "\n"
        << "| Number of streams in this thread: " << m_stream_packs.size() << "\n"
        << "+#############################################\n";
    return out;
}

void MediaSenderIONode::initialize_send_flows(const std::vector<TwoTupleFlow>& flows)
{
    std::vector<size_t> flows_per_stream(m_stream_packs.size(), 0);

    for (size_t flow = 0; flow < flows.size(); flow++) {
        flows_per_stream[flow % m_stream_packs.size()]++;
    }

    size_t flows_offset = 0;

    for (size_t strm_indx = 0; strm_indx < m_stream_packs.size(); strm_indx++) {
        m_stream_packs[strm_indx].flows = std::vector<TwoTupleFlow>(
            flows.begin() + flows_offset,
            flows.begin() + flows_offset + flows_per_stream[strm_indx]);
        flows_offset += flows_per_stream[strm_indx];
    }
}

void MediaSenderIONode::initialize_streams()
{
    // TODO: Update this when adding SDP parser.
    std::string sender_sdp = m_media_settings.sdp;
    constexpr size_t flow_index = 0;  // For now, there is one flow per stream.
    std::string destination_ip;
    uint16_t destination_port;
    std::string stream_sdp;
    size_t stream_idx = 0;

    for (auto& stream_pack : m_stream_packs) {
        stream_sdp = sender_sdp;
        destination_ip = stream_pack.flows[flow_index].get_ip();
        destination_port = stream_pack.flows[flow_index].get_port();

        // Update destination IP and port in the SDP file:
        replace_all(stream_sdp, "c=IN IP4 ", destination_ip, "/");
        replace_all(stream_sdp, "incl IN IP4 ", destination_ip, " ");
        replace_all(stream_sdp, "m=video ", std::to_string(destination_port), " ");

        auto network_address = TwoTupleFlow(
            stream_idx++,
            m_network_address.get_source_ip(),
            m_network_address.get_source_port());
        m_media_settings.sdp = stream_sdp;

        MediaStreamSettings stream_settings(network_address, m_media_settings,
            m_num_of_packets_in_chunk, m_packet_payload_size, m_data_stride_size,
            m_app_header_stride_size, m_dscp, m_pcp, m_ecn);

        stream_pack.stream = std::unique_ptr<RtpVideoSendStream>(
                new RtpVideoSendStream(stream_settings));
        stream_pack.buffer_writer = std::unique_ptr<RTPVideoMockBufferWriter>(
            new RTPVideoMockBufferWriter(stream_settings.m_media_settings,
                stream_settings.m_app_header_stride_size, stream_settings.m_data_stride_size,
                stream_settings.m_packet_payload_size,
                m_memory_utils.get_header_memory_utils(), m_memory_utils.get_payload_memory_utils()));
    }
    m_media_settings.sdp = std::move(sender_sdp);
}

ReturnStatus MediaSenderIONode::initialize_memory_layout()
{
    determine_memory_layout_for_single_block(m_block_header_memory_size, m_block_payload_memory_size);
    if (!m_video_file.empty() && !m_dynamic_video_file_load) {
        auto rc = get_number_of_mem_blocks_per_file(m_num_of_memory_blocks);
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to get number of memory blocks per file" << std::endl;
            return rc;
        }
    }

    m_header_total_memory_size = m_block_header_memory_size * m_num_of_memory_blocks * m_stream_packs.size();
    m_payload_total_memory_size = m_block_payload_memory_size * m_num_of_memory_blocks * m_stream_packs.size();
    return ReturnStatus::success;
}

ReturnStatus MediaSenderIONode::determine_memory_layout(HeaderPayloadMemoryLayoutRequest& memory_layout_request) const
{
    if (m_payload_total_memory_size == 0) {
        std::cerr << "No memory requirements found for sender " << get_index() << std::endl;
        return ReturnStatus::failure;
    }
    memory_layout_request.header_payload_buffers_size = {m_header_total_memory_size, m_payload_total_memory_size};
    return ReturnStatus::success;
}

bool MediaSenderIONode::is_internal_allocation_requested(const HeaderPayloadMemoryLayout& layout) const
{
    return layout.header_memory_size == 0 && layout.payload_memory_size == 0;
}

ReturnStatus MediaSenderIONode::apply_memory_layout(const HeaderPayloadMemoryLayoutResponse& memory_layout_response)
{
    if (is_internal_allocation_requested(memory_layout_response.memory_layout)) {
        return apply_memory_layout_to_subcomponents();
    }

    ReturnStatus status = validate_memory_layout(memory_layout_response);
    if (status != ReturnStatus::success) {
        std::cerr << "Invalid memory layout provided" << std::endl;
        return status;
    }
    return apply_memory_layout_to_subcomponents(memory_layout_response.memory_layout);
}

ReturnStatus MediaSenderIONode::validate_memory_layout(const HeaderPayloadMemoryLayoutResponse& memory_layout_response) const
{
    const auto& io_node_memory_layout = memory_layout_response.memory_layout;

    if (m_payload_total_memory_size == 0) {
        std::cerr << "No memory requirements found for sender " << get_index() << std::endl;
        return ReturnStatus::failure;
    }

    if ((m_header_total_memory_size > 0 && io_node_memory_layout.header_memory_ptr == nullptr) ||
        io_node_memory_layout.payload_memory_ptr == nullptr) {
        std::cerr << "Invalid memory layout provided" << std::endl;
        return ReturnStatus::failure;
    }

    if (io_node_memory_layout.header_memory_size < m_header_total_memory_size ||
        io_node_memory_layout.payload_memory_size < m_payload_total_memory_size) {
        std::cerr << "Insufficient memory provided" << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus MediaSenderIONode::initialize_mem_blockset(
    MediaStreamMemBlockset& mem_blockset, uint8_t* header_memory_ptr,
    uint8_t* payload_memory_ptr, const HeaderPayloadMemoryLayout& io_node_memory_layout,
    const std::string& video_file)
{
    std::ifstream input_file;
    input_file.open(video_file, std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Failed to open file: " << m_video_file << std::endl;
        return ReturnStatus::failure;
    }

    input_file.clear();
    input_file.seekg(0, std::ios::beg);
    size_t header_offset = 0;

    if (!is_hds_on()) {
        header_offset = m_media_settings.protocol_header_size;
    }

    ReturnStatus rc;
    for (size_t i = 0; i < m_num_of_memory_blocks; ++i) {
        rc = fill_memblock_from_file(payload_memory_ptr, m_block_payload_memory_size,
            header_offset, input_file);
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to read frames from file" << std::endl;
            input_file.close();
            return rc;
        }
        if (is_hds_on()) {
            mem_blockset.set_block_memory(i, 0, header_memory_ptr, m_block_header_memory_size,
                io_node_memory_layout.register_memory ? io_node_memory_layout.header_memory_keys[0] : RMX_MKEY_INVALID);
            mem_blockset.set_block_memory(i, 1, payload_memory_ptr, m_block_payload_memory_size,
                io_node_memory_layout.register_memory ? io_node_memory_layout.payload_memory_keys[0] : RMX_MKEY_INVALID);
            mem_blockset.set_block_layout(i, m_mem_block_payload_sizes.data(), m_mem_block_header_sizes.data());
            header_memory_ptr += m_block_header_memory_size;
        } else {
            mem_blockset.set_block_memory(i, 0, payload_memory_ptr, m_block_payload_memory_size,
                io_node_memory_layout.register_memory ? io_node_memory_layout.payload_memory_keys[0] : RMX_MKEY_INVALID);
            mem_blockset.set_block_layout(i, m_mem_block_payload_sizes.data(), nullptr);
        }
        payload_memory_ptr += m_block_payload_memory_size;
    }
    input_file.close();
    return ReturnStatus::success;
}

ReturnStatus MediaSenderIONode::initialize_mem_blockset(
    MediaStreamMemBlockset& mem_blockset, uint8_t* header_memory_ptr,
    uint8_t* payload_memory_ptr, const HeaderPayloadMemoryLayout& io_node_memory_layout)
{
    for (size_t i = 0; i < m_num_of_memory_blocks; ++i) {
        if (is_hds_on()) {
            mem_blockset.set_block_memory(i, 0, header_memory_ptr, m_block_header_memory_size,
                io_node_memory_layout.register_memory ? io_node_memory_layout.header_memory_keys[0] : RMX_MKEY_INVALID);
            mem_blockset.set_block_memory(i, 1, payload_memory_ptr, m_block_payload_memory_size,
                io_node_memory_layout.register_memory ? io_node_memory_layout.payload_memory_keys[0] : RMX_MKEY_INVALID);
            mem_blockset.set_block_layout(i, m_mem_block_payload_sizes.data(), m_mem_block_header_sizes.data());
            header_memory_ptr += m_block_header_memory_size;
        } else {
            mem_blockset.set_block_memory(i, 0, payload_memory_ptr, m_block_payload_memory_size,
                io_node_memory_layout.register_memory ? io_node_memory_layout.payload_memory_keys[0] : RMX_MKEY_INVALID);
            mem_blockset.set_block_layout(i, m_mem_block_payload_sizes.data(), nullptr);
        }
        payload_memory_ptr += m_block_payload_memory_size;
    }
    return ReturnStatus::success;
}

ReturnStatus MediaSenderIONode::initialize_mem_blockset(MediaStreamMemBlockset& mem_blockset)
{
    for (size_t i = 0; i < m_num_of_memory_blocks; ++i) {
        if (is_hds_on()) {
            mem_blockset.set_block_layout(i, m_mem_block_payload_sizes.data(), m_mem_block_header_sizes.data());
        } else {
            mem_blockset.set_block_layout(i, m_mem_block_payload_sizes.data(), nullptr);
        }
    }
    return ReturnStatus::success;
}

ReturnStatus MediaSenderIONode::apply_memory_layout_to_subcomponents()
{
    if (m_packet_header_size) {
        m_mem_block_header_sizes.resize(m_num_of_packets_in_mem_block, m_packet_header_size);
    }
    m_mem_block_payload_sizes.resize(m_num_of_packets_in_mem_block, m_packet_payload_size);

    for (auto& stream_pack : m_stream_packs) {
        stream_pack.mem_blockset = std::unique_ptr<MediaStreamMemBlockset>(
                new MediaStreamMemBlockset(m_num_of_memory_blocks,
                                           is_hds_on() ? 2 : 1,
                                           m_num_of_chunks_in_mem_block));
        stream_pack.mem_blockset->set_rivermax_to_allocate_memory();
        initialize_mem_blockset(*stream_pack.mem_blockset);
        MediaMemoryLayoutResponse memory_layout(*stream_pack.mem_blockset.get());
        ReturnStatus rc = stream_pack.stream->apply_memory_layout(memory_layout);
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to apply memory layout to stream (" << stream_pack.stream->get_id() << ")" << std::endl;
            return rc;
        }
    }
    return ReturnStatus::success;
}

ReturnStatus MediaSenderIONode::apply_memory_layout_to_subcomponents(
    const HeaderPayloadMemoryLayout& memory_layout)
{
    if (m_packet_header_size) {
        m_mem_block_header_sizes.resize(m_num_of_packets_in_mem_block, m_packet_header_size);
    }
    m_mem_block_payload_sizes.resize(m_num_of_packets_in_mem_block, m_packet_payload_size);

    uint8_t* header_memory_ptr = static_cast<uint8_t*>(memory_layout.header_memory_ptr);
    uint8_t* payload_memory_ptr = static_cast<uint8_t*>(memory_layout.payload_memory_ptr);

    size_t header_memory_for_stream_size = m_num_of_memory_blocks * m_block_header_memory_size;
    size_t payload_memory_for_stream_size = m_num_of_memory_blocks * m_block_payload_memory_size;

    for (auto& stream_pack : m_stream_packs) {
        stream_pack.mem_blockset = std::unique_ptr<MediaStreamMemBlockset>(
                new MediaStreamMemBlockset(m_num_of_memory_blocks,
                                           is_hds_on() ? 2 : 1,
                                           m_num_of_chunks_in_mem_block));
        ReturnStatus rc;
        if (!m_video_file.empty() && !m_dynamic_video_file_load) {
            rc = initialize_mem_blockset(*stream_pack.mem_blockset, header_memory_ptr,
                payload_memory_ptr, memory_layout, m_video_file);
        } else {
            rc = initialize_mem_blockset(*stream_pack.mem_blockset, header_memory_ptr,
                payload_memory_ptr, memory_layout);
        }
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to initialize memory blockset" << std::endl;
            return rc;
        }
        MediaMemoryLayoutResponse memory_layout_response(*stream_pack.mem_blockset.get());
        rc = stream_pack.stream->apply_memory_layout(memory_layout_response);
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to apply memory layout to stream (" << stream_pack.stream->get_id() << ")" << std::endl;
            return rc;
        }
        header_memory_ptr += header_memory_for_stream_size;
        payload_memory_ptr += payload_memory_for_stream_size;
    }
    return ReturnStatus::success;
}

void MediaSenderIONode::print_parameters()
{
    if (!m_print_parameters) {
        return;
    }

    std::stringstream sender_parameters;
    sender_parameters << this;
    for (auto& stream_pack : m_stream_packs) {
        sender_parameters << *stream_pack.stream;
    }
    std::cout << sender_parameters.str() << std::endl;
}

ReturnStatus MediaSenderIONode::process_frame()
{
    for (auto& stream_pack : m_stream_packs) {
        if (!stream_pack.frame_provider) {
            continue;
        }
        std::shared_ptr<MediaFrame> frame = stream_pack.frame_provider->get_frame_blocking();

        if (SignalHandler::get_received_signal() >= 0) {
            return ReturnStatus::success;
        }
        if (!frame) {
            continue;
        }
        auto rc = stream_pack.buffer_writer->set_next_frame(std::move(frame));
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to set next frame" << std::endl;
            return rc;
        }
    }

    return ReturnStatus::success;
}

void MediaSenderIONode::operator()()
{
    set_cpu_resources();
    ReturnStatus rc = create_streams();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to create sender (" << m_index << ") streams" << std::endl;
        return;
    }
    print_parameters();
    prepare_buffers();

    /*
    * Currently the logic in the sender is that all the streams start
    * in the same time and keep aligned during the run. It can be updated in the future.
    */
    uint64_t time_now_ns = get_time_now_ns();
    double send_time_ns = 0;
    for (auto& stream_pack : m_stream_packs) {
        send_time_ns = stream_pack.stream->calculate_send_time_ns(time_now_ns);
        stream_pack.buffer_writer->set_first_packet_timestamp(send_time_ns);
    }
    const double start_send_time_ns = send_time_ns;
    size_t sent_mem_block_counter = 0;
    auto get_send_time_ns = [&]() { return (
        start_send_time_ns
        + m_media_settings.frame_field_time_interval_ns
        * m_media_settings.frames_fields_in_mem_block
        * sent_mem_block_counter);
    };
    uint64_t commit_timestamp_ns = 0;
    size_t chunk_in_frame_counter;
    rc = ReturnStatus::success;
    auto first_chunk_in_frame = false;

    // Determine which function to use based on header data split mode:
    std::function<ReturnStatus(MediaStreamPack&)> write_buffer_callback;
    if (is_hds_on()) {
        write_buffer_callback = [](auto& stream_pack) {
            return stream_pack.buffer_writer->write_buffer(
                stream_pack.chunk_handler->get_app_hdr_ptr(),
                stream_pack.chunk_handler->get_data_ptr(),
                stream_pack.chunk_handler->get_length());
        };
    } else {
        write_buffer_callback = [](auto& stream_pack) {
            return stream_pack.buffer_writer->write_buffer(
                stream_pack.chunk_handler->get_data_ptr(),
                stream_pack.chunk_handler->get_length());
        };
    }

    while (likely(rc != ReturnStatus::failure && SignalHandler::get_received_signal() < 0)) {
        chunk_in_frame_counter = 0;
        send_time_ns = get_send_time_ns();
        wait_for_next_frame(static_cast<uint64_t>(send_time_ns));
        rc = process_frame();
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to process frame" << std::endl;
            break;
        }
        do {
            for (auto& stream_pack : m_stream_packs) {
                do {
                    rc = stream_pack.stream->blocking_get_next_chunk(*stream_pack.chunk_handler, BLOCKING_CHUNK_RETRIES);
                } while (unlikely(rc == ReturnStatus::no_free_chunks));
                if (unlikely(rc != ReturnStatus::success)) {
                    break;
                }
                write_buffer_callback(stream_pack);
                first_chunk_in_frame = unlikely(chunk_in_frame_counter % m_media_settings.chunks_in_frame_field == 0);
                commit_timestamp_ns = get_commit_timestamp_ns(first_chunk_in_frame, send_time_ns, stream_pack.stream->get_id());
                do {
                    rc = stream_pack.stream->blocking_commit_chunk(*stream_pack.chunk_handler,
                            commit_timestamp_ns, BLOCKING_CHUNK_RETRIES);
                } while (unlikely(rc == ReturnStatus::hw_send_queue_full));
                if (unlikely(rc != ReturnStatus::success)) {
                    break;
                }
            }
            if ((chunk_in_frame_counter % m_media_settings.chunks_in_frame_field) == 0) {
                send_time_ns += m_media_settings.frame_field_time_interval_ns;
            }
        } while (likely(rc == ReturnStatus::success &&
                        ++chunk_in_frame_counter < m_media_settings.chunks_in_frame_field));

        sent_mem_block_counter++;
    }

    rc = destroy_streams();
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to destroy sender (" << m_index << ") streams" << std::endl;
        return;
    }
}

ReturnStatus MediaSenderIONode::create_streams()
{
    ReturnStatus rc;

    for (auto& stream_pack : m_stream_packs) {
        rc = stream_pack.stream->create_stream();
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to create stream (" << stream_pack.stream->get_id() << ")" << std::endl;
            return rc;
        }
        stream_pack.chunk_handler = std::unique_ptr<MediaChunk>(
                new MediaChunk(stream_pack.stream->get_id(), m_num_of_packets_in_chunk,
                               stream_pack.stream->is_hds_on()));
    }

    return ReturnStatus::success;
}

ReturnStatus MediaSenderIONode::destroy_streams()
{
    ReturnStatus rc;

    for (auto& stream_pack : m_stream_packs) {

        rc = stream_pack.chunk_handler->cancel_unsent();
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to cancel media streams" << std::endl;
            return rc;
        }

        rc = stream_pack.stream->destroy_stream();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to destroy stream (" << stream_pack.stream->get_id() << ")" << std::endl;
            return rc;
        }
    }

    return ReturnStatus::success;
}

void MediaSenderIONode::set_cpu_resources()
{
    set_current_thread_affinity(m_cpu_core_affinity);
    rt_set_thread_priority(RMAX_THREAD_PRIORITY_TIME_CRITICAL - 1);
}

inline void MediaSenderIONode::prepare_buffers()
{
    // TODO: Add buffer preparation, for now, send random garbage as payload.
}

void MediaSenderIONode::wait_for_next_frame(uint64_t sleep_till_ns)
{
    uint64_t time_now_ns = get_time_now_ns();

    if (!m_sleep_between_operations || sleep_till_ns <= time_now_ns) {
        return;
    }

    size_t sleep_time_ns = sleep_till_ns - time_now_ns;

    if (sleep_time_ns <= SLEEP_THRESHOLD_NS) {
        return;
    }

    sleep_time_ns -= SLEEP_THRESHOLD_NS;
#ifdef __linux__
    std::this_thread::sleep_for(std::chrono::nanoseconds(sleep_time_ns));
#else
    sleep_till_ns -= sleep_time_ns;
    while (get_time_now_ns() < sleep_till_ns);
#endif
}

void MediaSenderIONode::determine_memory_layout_for_single_block(
    size_t& block_header_memory_size, size_t& block_payload_memory_size)
{
    size_t num_of_strides_per_mem_block = m_num_of_packets_in_chunk * m_num_of_chunks_in_mem_block;
    size_t payload_memory_size = m_data_stride_size * num_of_strides_per_mem_block;

    if (is_hds_on()) {
        block_header_memory_size = m_app_header_stride_size * num_of_strides_per_mem_block;
        block_payload_memory_size = payload_memory_size;
    } else {
        block_header_memory_size = 0;
        block_payload_memory_size = payload_memory_size;
    }
    if (m_gpu_enabled) {
        block_payload_memory_size = m_memory_utils.get_aligned_payload_size(block_payload_memory_size);
    }
}

ReturnStatus MediaSenderIONode::set_frame_provider(size_t stream_index,
    std::shared_ptr<IFrameProvider> frame_provider, MediaType media_type, bool contains_payload)
{
    if (frame_provider == nullptr) {
        std::cerr << "Invalid frame_provider" << std::endl;
        return ReturnStatus::failure;
    }
    if (stream_index >= m_stream_packs.size()) {
        std::cerr << "Invalid stream index " << stream_index << std::endl;
        return ReturnStatus::failure;
    }

    m_stream_packs[stream_index].frame_provider = std::move(frame_provider);
    std::unique_ptr<RTPMediaBufferWriter> buffer_writer = RTPMediaBufferWriter::get_rtp_media_buffer_writer(
        media_type, contains_payload, m_media_settings,
        m_app_header_stride_size, m_data_stride_size,
        m_packet_payload_size,
        m_memory_utils.get_header_memory_utils(), m_memory_utils.get_payload_memory_utils());
    if (buffer_writer) {
        m_stream_packs[stream_index].buffer_writer = std::move(buffer_writer);
    } else {
        std::cout << "Frame provider was set for stream " << stream_index <<
            " but buffer writer was not created" << std::endl;
    }
    return ReturnStatus::success;
}

ReturnStatus MediaSenderIONode::get_number_of_mem_blocks_per_file(size_t& number_of_mem_blocks) const
{
    number_of_mem_blocks = 0;
    std::ifstream input_file;
    input_file.open(m_video_file, std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Failed to open file: " << m_video_file << std::endl;
        return ReturnStatus::failure;
    }

    input_file.seekg(0, std::ios::end);
    size_t file_size = input_file.tellg();
    input_file.close();

    size_t mem_block_payload_size_in_bytes = 0;
    if (is_hds_on()) {
        mem_block_payload_size_in_bytes = m_packet_payload_size * m_media_settings.packets_in_frame_field *
        m_media_settings.frames_fields_in_mem_block;
    } else {
        mem_block_payload_size_in_bytes = (m_packet_payload_size - m_media_settings.protocol_header_size) *
        m_media_settings.packets_in_frame_field * m_media_settings.frames_fields_in_mem_block;
    }

    std::cout << "File size: " << file_size << " [bytes]" << std::endl;
    number_of_mem_blocks = file_size / mem_block_payload_size_in_bytes;
    std::cout << "Required Number of memory blocks: " << number_of_mem_blocks << std::endl;
    return ReturnStatus::success;
}

ReturnStatus MediaSenderIONode::fill_memblock_from_file(byte_t* block_memory_buffer, size_t block_memory_size,
    size_t header_offset, std::ifstream& input_file) const
{
    auto mem_utils = m_memory_utils.get_payload_memory_utils();
    if (!mem_utils) {
        std::cerr << "Failed to get memory utils" << std::endl;
        return ReturnStatus::failure;
    }
    if (!input_file.is_open()) {
        std::cerr << "File is not open."<< std::endl;
        return ReturnStatus::failure;
    }

    auto temp_buffer = std::make_unique<byte_t[]>(block_memory_size);
    size_t packet_size = m_packet_payload_size - header_offset;
    size_t mem_block_payload_size_in_bytes = packet_size * m_media_settings.packets_in_frame_field *
        m_media_settings.frames_fields_in_mem_block;
    size_t total_bytes_read = 0;
    byte_t* cur_data_ptr = temp_buffer.get() + header_offset;

    while (mem_block_payload_size_in_bytes > total_bytes_read) {
        input_file.read(reinterpret_cast<char*>(cur_data_ptr), packet_size);
        std::streamsize bytes_read = input_file.gcount();

        if (bytes_read == 0) {
            // If we reached EOF, break out normally.
            if (input_file.eof()) {
                break;
            }
            // If not EOF but still no data read, then an error occurred.
            if (input_file.fail() && !input_file.eof()) {
                return ReturnStatus::failure;
            }
        }

        if (bytes_read < static_cast<std::streamsize>(packet_size)) {
            break;
        }
        total_bytes_read += bytes_read;
        cur_data_ptr += m_data_stride_size;
    }
    mem_utils->memory_copy(block_memory_buffer, temp_buffer.get(), block_memory_size);

    return ReturnStatus::success;
}
