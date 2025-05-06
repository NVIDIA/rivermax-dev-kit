/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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

#include "senders/media_sender_io_node.h"
#include "api/rmax_apps_lib_api.h"
#include "services/utils/cpu.h"

using namespace ral::io_node;
using namespace ral::lib::core;
using namespace ral::lib::services;

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
        std::shared_ptr<MemoryUtils> mem_utils, time_handler_ns_cb_t time_hanlder_cb) :
    m_stream_packs(num_of_streams),
    m_media_settings(app_settings->media),
    m_index(index),
    m_network_address(network_address),
    m_sleep_between_operations(app_settings->sleep_between_operations),
    m_print_parameters(app_settings->print_parameters),
    m_cpu_core_affinity(cpu_core_affinity),
    m_hw_queue_full_sleep_us(app_settings->hw_queue_full_sleep_us),
    m_buffer_writer(nullptr),
    m_mem_utils(mem_utils),
    m_num_of_chunks_in_mem_block(app_settings->num_of_chunks_in_mem_block),
    m_packet_payload_size(app_settings->packet_payload_size),
    m_num_of_packets_in_chunk(app_settings->num_of_packets_in_chunk),
    m_num_of_packets_in_mem_block(app_settings->num_of_packets_in_mem_block),
    m_data_stride_size(align_up_pow2(m_packet_payload_size, get_cache_line_size())),
    m_dscp(0), m_pcp(0), m_ecn(0),
    m_get_time_ns_cb(time_hanlder_cb)
{
    m_stream_packs.resize(num_of_streams);
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
                m_num_of_packets_in_chunk, m_packet_payload_size, m_data_stride_size, 0,
                m_dscp, m_pcp, m_ecn);

        stream_pack.stream = std::unique_ptr<RtpVideoSendStream>(
                new RtpVideoSendStream(stream_settings, *stream_pack.mem_blockset.get()));
    }
    m_media_settings.sdp = sender_sdp;
}

void MediaSenderIONode::initialize_memory()
{
    distribute_memory_for_streams();
}

void MediaSenderIONode::distribute_memory_for_streams()
{
    m_mem_block_payload_sizes.resize(m_num_of_packets_in_mem_block, m_packet_payload_size);

    /* TODO - Remove this code and add the following:
     *   * Header data split
     *   * Application memory allocation
     *   * Create library buffer wrapper class to abstract the code blow.
    */

    for (auto& stream_pack : m_stream_packs) {
        stream_pack.mem_blockset = std::unique_ptr<MediaStreamMemBlockset>(
                new MediaStreamMemBlockset(1, 1, m_num_of_chunks_in_mem_block));
        stream_pack.mem_blockset->set_rivermax_to_allocate_memory();
        stream_pack.mem_blockset->set_block_layout(0, m_mem_block_payload_sizes.data(), nullptr);
    }
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

    while (likely(rc != ReturnStatus::failure && SignalHandler::get_received_signal() < 0)) {
        chunk_in_frame_counter = 0;
        send_time_ns = get_send_time_ns();
        wait_for_next_frame(static_cast<uint64_t>(send_time_ns));

        do {
            for (auto& stream_pack : m_stream_packs) {
                do {
                    rc = stream_pack.stream->blocking_get_next_chunk(*stream_pack.chunk_handler, BLOCKING_CHUNK_RETRIES);
                } while (unlikely(rc == ReturnStatus::no_free_chunks));
                if (unlikely(rc != ReturnStatus::success)) {
                    break;
                }

                stream_pack.stream->prepare_chunk_to_send(*stream_pack.chunk_handler.get());
                if (unlikely(chunk_in_frame_counter % m_media_settings.chunks_in_frame_field == 0)) {
                    commit_timestamp_ns = static_cast<uint64_t>(send_time_ns);
                } else {
                    commit_timestamp_ns = 0;
                }

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
                new MediaChunk(stream_pack.stream->get_id(), m_num_of_packets_in_chunk, 0)); // HDS is not supported
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
