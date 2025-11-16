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

#include <chrono>
#include <climits>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <rivermax_api.h>
#include <sstream>
#include <thread>
#include <vector>

#include "rt_threads.h"

#include "rdk/services/media/media_settings.h"
#include "rdk/io_node/senders/media_sender_io_node.h"
#include "rdk/services/cpu/cpu.h"
#include "rdk/services/error_handling/error_handling.h"
#include "rdk/services/media/media.h"

using namespace std::chrono;
using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

constexpr size_t MediaSenderIONode::DEFAULT_PRINT_TIME_INTERVAL_MS;

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
        const AppSettings& app_settings,
        const MediaSettings& media_settings,
        size_t index, size_t num_of_streams, int cpu_core_affinity,
        IONodeMemoryUtils& memory_utils,
        time_handler_ns_cb_t time_hanlder_cb) :
    m_stream_packs(num_of_streams),
    m_media_settings(media_settings),
    m_index(index),
    m_network_address(network_address),
    m_sleep_between_operations(app_settings.sleep_between_operations),
    m_print_parameters(app_settings.print_parameters),
    m_stats_report_interval_ms(app_settings.stats_report_interval_ms),
    m_stats_sent_media_unit_chunk_counter(0),
    m_cpu_core_affinity(cpu_core_affinity),
    m_hw_queue_full_sleep_us(app_settings.hw_queue_full_sleep_us),
    m_memory_utils(memory_utils),
    m_block_header_memory_size(0),
    m_block_payload_memory_size(0),
    m_header_total_memory_size(0),
    m_payload_total_memory_size(0),
    m_dscp(0), m_pcp(0), m_ecn(0),
    m_get_time_ns_cb(std::move(time_hanlder_cb)),
    m_gpu_enabled(app_settings.gpu_id != INVALID_GPU_ID),
    m_synchronizer(nullptr)
{
    m_stream_packs.resize(num_of_streams);
    initialize_memory_layout();
}

std::ostream& MediaSenderIONode::print(std::ostream& out) const
{
    out << "| Sender index: " << m_index << "\n"
        << "| Thread ID: 0x" << std::hex << std::this_thread::get_id() << std::dec << "\n"
        << "| CPU core affinity: " << m_cpu_core_affinity << "\n"
        << "| Number of streams in this thread: " << m_stream_packs.size() << "\n";
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

ReturnStatus MediaSenderIONode::initialize_streams()
{
    constexpr size_t flow_index = 0;  // For now, there is one flow per Tx stream.
    std::string destination_ip;
    uint16_t destination_port;
    size_t stream_idx = 0;
    SMPTEStandard smpte_standard = m_media_settings.get_smpte_standard();
    ReturnStatus rc;

    for (auto& stream_pack : m_stream_packs) {
        auto& destination_flow = stream_pack.flows[flow_index];
        destination_ip = destination_flow.get_ip();
        destination_port = destination_flow.get_port();

        auto network_address = TwoTupleFlow(stream_idx, m_network_address.get_source_ip(),
                                            m_network_address.get_source_port());

        MediaStreamSettings stream_settings(network_address, destination_flow, m_media_settings,
                                            m_dscp, m_pcp, m_ecn);

        stream_pack.stream = std::make_unique<MediaSendStream>(stream_settings);
        auto runtime_essence_provider = std::make_shared<NullEssenceProvider>(m_media_settings);
        auto preload_essence_provider = std::make_shared<NullEssenceProvider>(m_media_settings);

        rc = set_media_essence_providers(stream_idx, smpte_standard,
            std::move(preload_essence_provider), std::move(runtime_essence_provider), false);
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to set preload essence provider for stream " << stream_idx << std::endl;
            return rc;
        }
        if (!stream_pack.preload_packet_buffer_writer || !stream_pack.runtime_packet_buffer_writer) {
            std::cerr << "Failed to create preload or runtime packet buffer writer for stream " << stream_idx << std::endl;
            return ReturnStatus::failure;
        }

        stream_pack.number_of_memory_blocks = DEFAULT_NUMBER_OF_MEM_BLOCKS;

        m_header_total_memory_size += m_block_header_memory_size * stream_pack.number_of_memory_blocks;
        m_payload_total_memory_size += m_block_payload_memory_size * stream_pack.number_of_memory_blocks;
        stream_idx++;
    }
    return ReturnStatus::success;
}

ReturnStatus MediaSenderIONode::initialize_memory_layout()
{
    size_t num_of_strides_per_mem_block = m_media_settings.packets_in_chunk * m_media_settings.chunks_in_mem_block;
    size_t payload_memory_size = m_media_settings.data_stride_size * num_of_strides_per_mem_block;

    if (is_hds_on()) {
        m_block_header_memory_size = m_media_settings.app_header_stride_size * num_of_strides_per_mem_block;
        m_block_payload_memory_size = payload_memory_size;
    } else {
        m_block_header_memory_size = 0;
        m_block_payload_memory_size = payload_memory_size;
    }
    if (m_gpu_enabled) {
        m_block_payload_memory_size = m_memory_utils.get_aligned_payload_size(m_block_payload_memory_size);
    }
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

    ReturnStatus rc = validate_memory_layout(memory_layout_response);
    if (rc != ReturnStatus::success) {
        std::cerr << "Invalid memory layout provided" << std::endl;
        return rc;
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
    uint8_t* payload_memory_ptr, const HeaderPayloadMemoryLayout& io_node_memory_layout, size_t number_of_memory_blocks)
{
    // Provide size arrays only if it contains valid information. Otherwise, for dynamic streams, no sizes are provided.
    // Header sizes will also be nullptr for non-HDS streams.
    uint16_t* payload_sizes = m_mem_block_payload_sizes.empty() ? nullptr : m_mem_block_payload_sizes.data();
    uint16_t* header_sizes = m_mem_block_header_sizes.empty() ? nullptr : m_mem_block_header_sizes.data();

    for (size_t i = 0; i < number_of_memory_blocks; ++i) {
        if (is_hds_on()) {
            mem_blockset.set_block_memory(i, 0, header_memory_ptr, m_block_header_memory_size,
                io_node_memory_layout.register_memory ? io_node_memory_layout.header_memory_keys[0] : RMX_MKEY_INVALID);
            mem_blockset.set_block_memory(i, 1, payload_memory_ptr, m_block_payload_memory_size,
                io_node_memory_layout.register_memory ? io_node_memory_layout.payload_memory_keys[0] : RMX_MKEY_INVALID);
            header_memory_ptr += m_block_header_memory_size;
        } else {
            mem_blockset.set_block_memory(i, 0, payload_memory_ptr, m_block_payload_memory_size,
                io_node_memory_layout.register_memory ? io_node_memory_layout.payload_memory_keys[0] : RMX_MKEY_INVALID);
        }
        mem_blockset.set_block_layout(i, payload_sizes, header_sizes);
        payload_memory_ptr += m_block_payload_memory_size;
    }
    return ReturnStatus::success;
}

ReturnStatus MediaSenderIONode::initialize_mem_blockset(MediaStreamMemBlockset& mem_blockset, size_t number_of_memory_blocks)
{
    // Provide size arrays only if it contains valid information. Otherwise, for dynamic streams, no sizes are provided.
    // Header sizes will also be nullptr for non-HDS streams.
    uint16_t* payload_sizes = m_mem_block_payload_sizes.empty() ? nullptr : m_mem_block_payload_sizes.data();
    uint16_t* header_sizes = m_mem_block_header_sizes.empty() ? nullptr : m_mem_block_header_sizes.data();
    for (size_t i = 0; i < number_of_memory_blocks; ++i) {
        mem_blockset.set_block_layout(i, payload_sizes, header_sizes);
    }
    return ReturnStatus::success;
}

ReturnStatus MediaSenderIONode::apply_memory_layout_to_subcomponents()
{
    // Dynamic streams (ancillary) provides sizes at runtime. Otherwise, pre-fill size arrays.
    if (!m_media_settings.needs_dynamic_packet_sizes()) {
        if (m_media_settings.packet_app_header_size) {
            m_mem_block_header_sizes.resize(m_media_settings.packets_in_mem_block, m_media_settings.packet_app_header_size);
        }
        m_mem_block_payload_sizes.resize(m_media_settings.packets_in_mem_block, m_media_settings.packet_payload_size);
    }

    for (auto& stream_pack : m_stream_packs) {
        stream_pack.mem_blockset = std::unique_ptr<MediaStreamMemBlockset>(
                new MediaStreamMemBlockset(stream_pack.number_of_memory_blocks,
                                           is_hds_on() ? 2 : 1,
                                           m_media_settings.chunks_in_mem_block));
        stream_pack.mem_blockset->set_rivermax_to_allocate_memory();
        initialize_mem_blockset(*stream_pack.mem_blockset, stream_pack.number_of_memory_blocks);
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
    // Dynamic streams (ancillary) provides sizes at runtime. Otherwise, pre-fill size arrays.
    if (!m_media_settings.needs_dynamic_packet_sizes()) {
        if (m_media_settings.packet_app_header_size) {
            m_mem_block_header_sizes.resize(m_media_settings.packets_in_mem_block, m_media_settings.packet_app_header_size);
        }
        m_mem_block_payload_sizes.resize(m_media_settings.packets_in_mem_block, m_media_settings.packet_payload_size);
    }

    uint8_t* header_memory_ptr = static_cast<uint8_t*>(memory_layout.header_memory_ptr);
    uint8_t* payload_memory_ptr = static_cast<uint8_t*>(memory_layout.payload_memory_ptr);

    for (auto& stream_pack : m_stream_packs) {
        stream_pack.mem_blockset = std::unique_ptr<MediaStreamMemBlockset>(
                new MediaStreamMemBlockset(stream_pack.number_of_memory_blocks,
                                           is_hds_on() ? 2 : 1,
                                           m_media_settings.chunks_in_mem_block));
        ReturnStatus rc;
        rc = initialize_mem_blockset(*stream_pack.mem_blockset, header_memory_ptr,
            payload_memory_ptr, memory_layout, stream_pack.number_of_memory_blocks);
        stream_pack.header_memory_ptr = header_memory_ptr;
        stream_pack.payload_memory_ptr = payload_memory_ptr;
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
        header_memory_ptr += stream_pack.number_of_memory_blocks * m_block_header_memory_size;
        payload_memory_ptr += stream_pack.number_of_memory_blocks * m_block_payload_memory_size;
    }
    return ReturnStatus::success;
}

void MediaSenderIONode::print_parameters()
{
    if (!m_print_parameters) {
        return;
    }

    std::stringstream sender_parameters;
    sender_parameters << "+#############################################\n";
    sender_parameters << this;
    sender_parameters << "+---------------------------------------------\n";
    for (auto& stream_pack : m_stream_packs) {
        sender_parameters << *stream_pack.stream;
        sender_parameters << "+---------------------------------------------\n";
    }
    std::cout << sender_parameters.str() << std::endl;
}

ReturnStatus MediaSenderIONode::process_media_unit()
{
    for (auto& stream_pack : m_stream_packs) {
        if (!stream_pack.runtime_essence_provider) {
            continue;
        }
        std::shared_ptr<MediaUnit> media_unit = stream_pack.runtime_essence_provider->get_media_unit_blocking();

        if (SignalHandler::get_received_signal() >= 0) {
            return ReturnStatus::success;
        }
        if (!media_unit) {
            continue;
        }
        auto rc = stream_pack.runtime_packet_buffer_writer->set_next_media_unit(std::move(media_unit));
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to set next media unit" << std::endl;
            return rc;
        }
    }

    return ReturnStatus::success;
}

ReturnStatus MediaSenderIONode::coordinate_start_time(uint64_t& send_time_ns)
{
    if (!m_synchronizer) {
        return ReturnStatus::success;
    }

    uint64_t initial_requested_time_ns = send_time_ns;
    auto start_time_checker = [&](uint64_t proposed_time_ns) {
        /**
         * @brief: Align proposed time to interval boundaries
         *
         * Example: interval_ns = 1000 nanoseconds
         *
         * Timeline showing interval boundaries:
         *
         *     0         1000        2000        3000        4000
         *     |-----------|-----------|-----------|-----------|
         *     START     BOUNDARY    BOUNDARY    BOUNDARY    BOUNDARY
         *
         *
         * CASE 1: Proposed time is very close to a boundary (ACCEPTS)
         *
         *     0         1000        2000        3000
         *     |-----------|-----------|-----------|
         *                             ^ proposed at 1999 or 2001
         *                             (within margin of boundary 2000)
         *     Result: Return 0 (no adjustment needed)
         *
         *
         * CASE 2: Proposed time is in the middle (REJECTS - needs adjustment)
         *
         *     0         1000        2000        3000
         *     |-----------|-----------|-----------|
         *                      ^      ^
         *                   proposed  next boundary
         *                    at 1500  at 2000
         *                      |------|
         *                   adjustment = 500ns
         *
         *     Result: Return 500 (push forward 500ns to reach boundary 2000)
         *
         *
         * This logic ensures timing happens at interval boundaries with a small margin of error.
         */
        uint64_t interval_ns = static_cast<uint64_t>(m_media_settings.media_unit_time_interval_ns);

        // Calculate how far the proposition is from requested time.
        int64_t offset_from_initial_ns = static_cast<int64_t>(proposed_time_ns) - static_cast<int64_t>(initial_requested_time_ns);

        // Calculate the offset from the nearest interval boundary.
        int64_t interval_offset_ns = offset_from_initial_ns % interval_ns;

        // If it's 1us or less from nearest boundary, no adjustment needed.
        if (interval_offset_ns < NS_IN_USEC || (interval_ns - interval_offset_ns) < NS_IN_USEC) {
            return 0;
        }

        // Otherwise, calculate the offset to the next interval boundary.
        int64_t next_interval_offset_ns = ((offset_from_initial_ns / interval_ns) + 1) * interval_ns;
        uint64_t new_start_time_ns = initial_requested_time_ns + next_interval_offset_ns;

        // Return the adjustment needed to align to the next interval boundary.
        int adjustment_ns = static_cast<int>(new_start_time_ns - proposed_time_ns);
        return adjustment_ns;
    };

    uint64_t coordinated_start_time_ns = 0;
    ReturnStatus rc = m_synchronizer->request(send_time_ns, start_time_checker, coordinated_start_time_ns);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to request coordinated start time" << std::endl;
        return rc;
    }

    if (coordinated_start_time_ns != initial_requested_time_ns) {
        int64_t time_adjustment_ns = static_cast<int64_t>(coordinated_start_time_ns - initial_requested_time_ns);
        uint64_t interval_ns = static_cast<uint64_t>(m_media_settings.media_unit_time_interval_ns);
        int64_t unit_adjustment = time_adjustment_ns / interval_ns;

        std::string stream_name = m_media_settings.media_settings_calculator->get_smpte_standard_name();
        std::ostringstream oss;
        oss << "[Time Coordination] Sender[" << m_index << "] " << stream_name << ": adjusted "
            << (time_adjustment_ns / 1e6) << " ms"
            << " (" << unit_adjustment << " units)" << std::endl;
        std::cout << oss.str() << std::flush;
    }

    send_time_ns = coordinated_start_time_ns;
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
    preload_media_data();

    /*
    * Currently the logic in the sender is that all the streams start
    * in the same time and keep aligned during the run. It can be updated in the future.
    */
    uint64_t time_now_ns = get_time_now_ns();
    uint64_t desired_start_time_ns = time_now_ns + DEFAULT_STREAM_START_OFFSET_NS;
    double aligned_time_ns = m_media_settings.media_settings_calculator->align_time_to_media_unit_boundary_ns(desired_start_time_ns);
    uint64_t send_time_ns = static_cast<uint64_t>(aligned_time_ns);

    rc = coordinate_start_time(send_time_ns);
    if (rc != ReturnStatus::success) {
        return;
    }

    double start_send_time_ns = static_cast<double>(send_time_ns);
    double transmit_offset_ns = m_media_settings.media_settings_calculator->get_transmit_offset_ns();
    start_send_time_ns += transmit_offset_ns;

    for (auto& stream_pack : m_stream_packs) {
        stream_pack.runtime_packet_buffer_writer->set_initial_timestamp(start_send_time_ns);
    }

    size_t sent_field_counter = 0;
    size_t sent_field_counter_diff = 0;
    auto get_send_time_ns = [&]() { return (
        start_send_time_ns
        + m_media_settings.media_unit_time_interval_ns
        * sent_field_counter);
    };
    uint64_t commit_timestamp_ns = 0;
    size_t chunk_in_media_unit_counter;
    rc = ReturnStatus::success;
    auto first_chunk_in_media_unit = false;

    // Determine which function to use based on header data split mode:
    std::function<ReturnStatus(MediaStreamPack&)> write_buffer_callback;
    if (is_hds_on()) {
        write_buffer_callback = [](auto& stream_pack) {
            return stream_pack.runtime_packet_buffer_writer->write_buffer(
                stream_pack.chunk_handler->get_app_hdr_ptr(),
                stream_pack.chunk_handler->get_data_ptr(),
                stream_pack.chunk_handler->get_length(),
                stream_pack.chunk_handler->get_app_hdr_sizes_array(),
                stream_pack.chunk_handler->get_data_sizes_array());
        };
    } else {
        write_buffer_callback = [](auto& stream_pack) {
            return stream_pack.runtime_packet_buffer_writer->write_buffer(
                stream_pack.chunk_handler->get_data_ptr(),
                stream_pack.chunk_handler->get_length(),
                stream_pack.chunk_handler->get_data_sizes_array());
        };
    }

    auto stats_start_time = high_resolution_clock::now();
    while (likely(rc != ReturnStatus::failure && SignalHandler::get_received_signal() < 0)) {
        chunk_in_media_unit_counter = 0;
        send_time_ns = get_send_time_ns();
        wait_for_next_media_unit(static_cast<uint64_t>(send_time_ns));
        rc = process_media_unit();
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to process media unit" << std::endl;
            break;
        }
        do {
            for (auto& stream_pack : m_stream_packs) {
                size_t num_packets_for_chunk = stream_pack.runtime_packet_buffer_writer->get_num_packets_for_next_chunk();
                stream_pack.chunk_handler->set_length(num_packets_for_chunk);
                do {
                    rc = stream_pack.stream->blocking_get_next_chunk(*stream_pack.chunk_handler, BLOCKING_CHUNK_RETRIES);
                } while (unlikely(rc == ReturnStatus::no_free_chunks));
                if (unlikely(rc != ReturnStatus::success)) {
                    break;
                }
                write_buffer_callback(stream_pack);
                first_chunk_in_media_unit = unlikely(chunk_in_media_unit_counter % m_media_settings.chunks_in_media_unit == 0);
                commit_timestamp_ns = get_commit_timestamp_ns(first_chunk_in_media_unit, send_time_ns, stream_pack.stream->get_id());
                do {
                    rc = stream_pack.stream->blocking_commit_chunk(*stream_pack.chunk_handler,
                            commit_timestamp_ns, BLOCKING_CHUNK_RETRIES);
                } while (unlikely(rc == ReturnStatus::hw_send_queue_full));
                if (unlikely(rc != ReturnStatus::success)) {
                    break;
                }
            }
            if ((chunk_in_media_unit_counter % m_media_settings.chunks_in_media_unit) == 0) {
                send_time_ns += m_media_settings.media_unit_time_interval_ns;
            }
        } while (likely(rc == ReturnStatus::success &&
                        ++chunk_in_media_unit_counter < m_media_settings.chunks_in_media_unit));

        sent_field_counter++;
        m_stats_sent_media_unit_chunk_counter++;

        if (m_stats_report_interval_ms > 0) {
            auto now = high_resolution_clock::now();
            auto duration = now - stats_start_time;
            if (duration >= milliseconds{m_stats_report_interval_ms}) {
                print_statistics(std::cout, duration);
                reset_statistics();
                stats_start_time = now;
            }
        }
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
                new MediaChunk(stream_pack.stream->get_id(), m_media_settings.packets_in_chunk,
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

inline void MediaSenderIONode::preload_media_data()
{
    for (auto& stream_pack : m_stream_packs) {
        // Skip if no external memory (internal allocation will be filled by Rivermax)
        if (stream_pack.header_memory_ptr == nullptr && stream_pack.payload_memory_ptr == nullptr) {
            continue;
        }
        // Skip if no preload essence provider
        if (!stream_pack.preload_essence_provider || !stream_pack.preload_packet_buffer_writer) {
            continue;
        }

        for (size_t block_idx = 0; block_idx < stream_pack.number_of_memory_blocks; ++block_idx) {
            byte_t* block_header_ptr = is_hds_on() ?
                stream_pack.header_memory_ptr + (block_idx * m_block_header_memory_size) : nullptr;
            byte_t* block_payload_ptr = stream_pack.payload_memory_ptr + (block_idx * m_block_payload_memory_size);

            // Process all media units for this block
            for (size_t unit_idx = 0; unit_idx < m_media_settings.media_units_in_mem_block; ++unit_idx) {
                std::shared_ptr<MediaUnit> media_unit = stream_pack.preload_essence_provider->get_media_unit_blocking();
                if (!media_unit) {
                    std::cerr << "Failed to get media unit." << std::endl;
                    break;
                }

                auto rc = stream_pack.preload_packet_buffer_writer->set_next_media_unit(std::move(media_unit));
                if (rc != ReturnStatus::success) {
                    std::cerr << "Failed to set media unit." << block_idx << std::endl;
                    break;
                }

                if (is_hds_on()) {
                    rc = stream_pack.preload_packet_buffer_writer->write_buffer(
                        block_header_ptr,
                        block_payload_ptr,
                        m_media_settings.packets_in_media_unit);
                } else {
                    rc = stream_pack.preload_packet_buffer_writer->write_buffer(
                        block_payload_ptr,
                        m_media_settings.packets_in_media_unit);
                }

                if (rc != ReturnStatus::success) {
                    std::cerr << "Failed to write to buffer" << std::endl;
                }

                block_payload_ptr += m_media_settings.packets_in_media_unit * m_media_settings.data_stride_size;
                block_header_ptr += is_hds_on() ?
                    m_media_settings.packets_in_media_unit * m_media_settings.app_header_stride_size : 0;
            }
        }
    }
}

void MediaSenderIONode::wait_for_next_media_unit(uint64_t sleep_till_ns)
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

size_t MediaSenderIONode::calculate_required_memory_blocks(size_t essence_size) const
{
    if (essence_size == 0) {
        return DEFAULT_NUMBER_OF_MEM_BLOCKS;
    }
    size_t required_number_of_memory_blocks = essence_size / m_block_payload_memory_size;
    return required_number_of_memory_blocks;
}

ReturnStatus MediaSenderIONode::set_media_essence_providers(
    size_t stream_index, SMPTEStandard smpte_standard,
    std::shared_ptr<IMediaEssenceProvider> preload_essence_provider,
    std::shared_ptr<IMediaEssenceProvider> runtime_essence_provider,
    bool runtime_contains_payload)
{
    if (runtime_essence_provider == nullptr && preload_essence_provider == nullptr) {
        std::cerr << "Invalid media essence provider" << std::endl;
        return ReturnStatus::failure;
    }
    if (stream_index >= m_stream_packs.size()) {
        std::cerr << "Invalid stream index " << stream_index << std::endl;
        return ReturnStatus::failure;
    }

    // Set runtime essence provider if provided
    if (runtime_essence_provider) {
        std::unique_ptr<IULPPacketBufferWriter> runtime_packet_buffer_writer =
            create_rtp_media_packet_buffer_writer(
                smpte_standard, runtime_contains_payload, m_media_settings,
                m_memory_utils.get_header_memory_utils(), m_memory_utils.get_payload_memory_utils());
        if (!runtime_packet_buffer_writer) {
            std::cerr << "Failed to create packet buffer writer" << std::endl;
            return ReturnStatus::failure;
        }
        m_stream_packs[stream_index].runtime_essence_provider = std::move(runtime_essence_provider);
        m_stream_packs[stream_index].runtime_packet_buffer_writer = std::move(runtime_packet_buffer_writer);
    }
    if (preload_essence_provider) {
        std::unique_ptr<IULPPacketBufferWriter> preload_packet_buffer_writer =
            create_rtp_media_packet_buffer_writer(
                smpte_standard, true, m_media_settings,
                m_memory_utils.get_header_memory_utils(), m_memory_utils.get_payload_memory_utils());
        if (!preload_packet_buffer_writer) {
            std::cerr << "Failed to create packet buffer writer" << std::endl;
            return ReturnStatus::failure;
        }
        m_stream_packs[stream_index].preload_essence_provider = std::move(preload_essence_provider);
        m_stream_packs[stream_index].preload_packet_buffer_writer = std::move(preload_packet_buffer_writer);
        // Update memory requirements based on preload essence size
        size_t essence_size;
        ReturnStatus rc = m_stream_packs[stream_index].preload_essence_provider->get_data_size(essence_size);
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to get data size for preload essence provider" << std::endl;
            return ReturnStatus::failure;
        }
        size_t required_number_of_memory_blocks = calculate_required_memory_blocks(essence_size);
        size_t diff_number_of_memory_blocks = required_number_of_memory_blocks - m_stream_packs[stream_index].number_of_memory_blocks;
        m_header_total_memory_size += m_block_header_memory_size * diff_number_of_memory_blocks;
        m_payload_total_memory_size += m_block_payload_memory_size * diff_number_of_memory_blocks;
        m_stream_packs[stream_index].number_of_memory_blocks = required_number_of_memory_blocks;
    }
    return ReturnStatus::success;
}

void MediaSenderIONode::print_statistics(
    std::ostream& out, const std::chrono::high_resolution_clock::duration& interval_duration) const
{
    uint64_t bytes_sent = (m_stats_sent_media_unit_chunk_counter) * m_media_settings.packets_in_media_unit * m_stream_packs.size() *
        (m_media_settings.packet_app_header_size + m_media_settings.packet_payload_size + RTP_ST_2110_20_SINGLE_SRD_HEADER_SIZE);
    float mbps = (bytes_sent * CHAR_BIT) / duration_cast<duration<float, std::micro>>(interval_duration).count();
    std::ostringstream oss;
    oss << " Sender: " << std::setw(3) << m_index
        << "  Streams: " << std::setw(3) << m_stream_packs.size()
        << "  Type: " << std::setw(11) << std::left << m_media_settings.media_settings_calculator->get_smpte_standard_name()
        << "  Media units sent: " << std::setw(3) << std::right << m_stats_sent_media_unit_chunk_counter
        << "  Bytes sent: " << std::setw(11) << bytes_sent
        << "  BW: " << std::setw(10) << std::fixed << std::setprecision(3) << mbps << " Mbps" << std::endl;
    out << oss.str();
}

void MediaSenderIONode::reset_statistics()
{
    m_stats_sent_media_unit_chunk_counter = 0;
}
