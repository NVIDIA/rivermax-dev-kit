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
#include <chrono>

#include <rivermax_api.h>
#include <rt_threads.h>

#include "rdk/io_node/misc/media_latency_io_node.h"
#include "rdk/services/error_handling/error_handling.h"
#include "rdk/services/media/media.h"
#include "rdk/services/memory_allocation/memory_allocation.h"
#include "rdk/core/stream/send/generic_stream.h"

using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

static constexpr size_t MEDIA_TX_REPLY_SIZE = 1200;
static constexpr uint64_t WAIT_SERVER_RECEIVE_NSEC = std::chrono::nanoseconds{ std::chrono::milliseconds{ 10 } }.count();
static constexpr uint64_t WAIT_SERVER_INACT_NSEC = std::chrono::nanoseconds{ std::chrono::milliseconds{ 20 } }.count();
static constexpr int WAIT_SERVER_REPLY_USEC = std::chrono::microseconds{ std::chrono::milliseconds{ 100 } }.count();
static constexpr size_t LAST_CHUNKS_SKIP_NUM = 1;

MediaTxIONode::MediaTxIONode(
        const LatencyNodeSettings& settings,
        std::shared_ptr<MemoryUtils> header_mem_utils,
        std::shared_ptr<MemoryUtils> payload_mem_utils,
        time_handler_ns_cb_t time_handler_cb
    ) : LatencyIONode(settings, std::move(header_mem_utils), std::move(payload_mem_utils), std::move(time_handler_cb)),
    m_app_settings(settings.app),
    m_receive_dim(StreamDimensions(DEFAULT_NUM_OF_RECEIVE_CHUNKS, 1, 0, MEDIA_TX_REPLY_SIZE)),
    m_hw_queue_full_sleep_us(settings.app->hw_queue_full_sleep_us),
    m_send_data_stride_size(0),
    m_send_header_stride_size(0),
    m_start_send_time_ns(0),
    m_marked_token(0),
    m_handled_token(0),
    m_trs(0)
{
}

void MediaTxIONode::initialize_send_stream()
{
    m_app_settings->num_of_total_streams = 1;
    m_app_settings->num_of_memory_blocks = 1;
    m_app_settings->media.frames_fields_in_mem_block = 2;
    initialize_media_settings(*m_app_settings);
    m_app_settings->num_of_chunks = m_app_settings->num_of_chunks_in_mem_block *
                                    m_app_settings->num_of_memory_blocks;
    if (m_gpu_direct_tx && (m_app_settings->packet_app_header_size == 0)) {
        m_app_settings->packet_app_header_size = RTP_HEADER_SIZE;
        m_app_settings->packet_payload_size -= RTP_HEADER_SIZE;
    }
    m_send_data_stride_size = align_up_pow2(m_app_settings->packet_payload_size,
                                            get_cache_line_size());
    m_send_header_stride_size = align_up_pow2(m_app_settings->packet_app_header_size,
                                            get_cache_line_size());
    auto network_address = TwoTupleFlow(0, m_network_address.get_source_ip(),
                                        m_network_address.get_source_port());

    MediaStreamSettings stream_settings(network_address, m_app_settings->media,
            m_app_settings->num_of_packets_in_chunk, m_app_settings->packet_payload_size,
            m_send_data_stride_size, m_send_header_stride_size);
    m_send_stream = std::shared_ptr<RtpVideoSendStream>(new RtpVideoSendStream(stream_settings));
}

ReturnStatus MediaTxIONode::query_memory_size(size_t& tx_header_size, size_t& tx_payload_size,
                                              size_t& rx_header_size, size_t& rx_payload_size)
{
    if (m_receive_stream->query_buffer_size(rx_header_size, rx_payload_size) != ReturnStatus::success) {
        return ReturnStatus::failure;
    }

    tx_header_size = m_send_header_stride_size * m_app_settings->num_of_packets_in_chunk *
                     m_app_settings->num_of_chunks;
    tx_payload_size = m_send_data_stride_size * m_app_settings->num_of_packets_in_chunk *
                     m_app_settings->num_of_chunks;
    return ReturnStatus::success;
};

void MediaTxIONode::distribute_memory_for_streams(rmx_mem_region& tx_header_mreg,
                                                  rmx_mem_region& tx_payload_mreg,
                                                  rmx_mem_region& rx_header_mreg,
                                                  rmx_mem_region& rx_payload_mreg)
{
    m_send_header_region = tx_header_mreg;
    m_send_payload_region = tx_payload_mreg;
    m_receive_header_region = rx_header_mreg;
    m_receive_payload_region = rx_payload_mreg;
    m_receive_stream->set_buffers(reinterpret_cast<byte_t*>(rx_header_mreg.addr),
                                  reinterpret_cast<byte_t*>(rx_payload_mreg.addr));
    m_send_mem_blockset = std::unique_ptr<MediaStreamMemBlockset>(
            new MediaStreamMemBlockset(1,
                                       m_app_settings->packet_app_header_size == 0 ? 1 : 2,
                                       m_app_settings->num_of_chunks_in_mem_block));

    if (m_app_settings->packet_app_header_size) {
        m_send_mem_blockset->set_block_memory(0, 0, tx_header_mreg.addr, tx_header_mreg.length,
                                              tx_payload_mreg.mkey);
        m_send_mem_blockset->set_block_memory(0, 1, tx_payload_mreg.addr, tx_payload_mreg.length,
                                              tx_payload_mreg.mkey);
        m_send_block_header_sizes.resize(m_app_settings->num_of_packets_in_mem_block,
                                      m_app_settings->packet_app_header_size);
    } else {
        m_send_mem_blockset->set_block_memory(0, 0, tx_payload_mreg.addr, tx_payload_mreg.length,
                                              tx_payload_mreg.mkey);
    }
    m_send_block_payload_sizes.resize(m_app_settings->num_of_packets_in_mem_block,
                                      m_app_settings->packet_payload_size);
    m_send_mem_blockset->set_block_layout(0, m_send_block_payload_sizes.data(),
                                          m_app_settings->packet_app_header_size ?
                                          m_send_block_header_sizes.data() : nullptr);
    m_send_stream->assign_memory_blocks(*m_send_mem_blockset.get());
}

void MediaTxIONode::print_parameters()
{
    if (!m_print_parameters) {
        return;
    }
    std::stringstream text_parameters;
    text_parameters << this;
    text_parameters << *m_send_stream;
    text_parameters << *m_receive_stream;
    std::cout << text_parameters.str() << std::endl;
}

std::ostream& MediaTxIONode::print(std::ostream& out) const
{
    return out;
};

ReturnStatus MediaTxIONode::create_send_stream()
{
    ReturnStatus rc;

    rc = m_send_stream->create_stream();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to create send stream" << std::endl;
        return rc;
    }

    return ReturnStatus::success;
}

ReturnStatus MediaTxIONode::destroy_send_stream()
{
    ReturnStatus rc;

    rc = m_send_stream->destroy_stream();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to destroy send stream" << std::endl;
        return rc;
    }

    return ReturnStatus::success;
}

void MediaTxIONode::initialize_receive_stream(const TwoTupleFlow& flow)
{
    ReceiveStreamSettings settings(flow,
        RMX_INPUT_APP_PROTOCOL_PACKET,
        RMX_INPUT_TIMESTAMP_SYNCED,
        {RMX_INPUT_STREAM_CREATE_INFO_PER_PACKET},
        m_receive_dim.num_of_chunks * m_receive_dim.num_of_packets_in_chunk,
        m_receive_dim.payload_size, 0);

    m_receive_stream = std::shared_ptr<ReceiveStream>(new ReceiveStream(settings));
}

ReturnStatus MediaTxIONode::create_receive_stream()
{
    ReturnStatus rc = m_receive_stream->create_stream();
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to create response stream" << std::endl;
    }
    return rc;
}

ReturnStatus MediaTxIONode::attach_receive_flow()
{
    ReturnStatus rc = m_receive_stream->attach_flow(m_receive_flow);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to attach flow to response stream" << std::endl;
    }
    return rc;
}

ReturnStatus MediaTxIONode::detach_receive_flow()
{
    ReturnStatus rc = m_receive_stream->detach_flow(m_receive_flow);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to detach flow from response stream" << std::endl;
    }
    return rc;
}

ReturnStatus MediaTxIONode::destroy_receive_stream()
{
    ReturnStatus rc = m_receive_stream->destroy_stream();
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to create response stream" << std::endl;
    }
    return rc;
}

void MediaTxIONode::wait_for_next_frame(uint64_t sleep_till_ns)
{
    uint64_t time_now_ns = get_time_now_ns();

    if (sleep_till_ns <= time_now_ns) {
        return;
    }

    size_t sleep_time_ns = sleep_till_ns - time_now_ns;

#ifdef __linux__
    if (m_app_settings->sleep_between_operations) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(sleep_time_ns));
    } else {
        while (get_time_now_ns() < sleep_till_ns);
    }
#else
    while (get_time_now_ns() < sleep_till_ns);
#endif
}

void MediaTxIONode::send_receive()
{
    ReturnStatus rc;
    m_chunk_handler = std::unique_ptr<MediaChunk>(
            new MediaChunk(m_send_stream->get_id(),
                           m_app_settings->num_of_packets_in_chunk,
                           m_send_stream->is_hds_on()));
    ReceiveChunk receive_chunk(m_receive_stream->get_id(), false);
    m_receive_stream->set_completion_moderation(1, 1, WAIT_SERVER_REPLY_USEC);
    uint64_t start_time_ns = get_time_now_ns();
    m_trs = m_send_stream->calculate_trs();
    m_start_send_time_ns = m_send_stream->calculate_send_time_ns(start_time_ns);
    size_t committed_frame_field_counter = 0;
    size_t completed_frame_field_counter = 0;
    auto& media_settings = m_app_settings->media;

    /* scheduled send time of the first packet of the field that will be committed next */
    auto get_send_time_of_next_field_ns = [&]() { return (
        m_start_send_time_ns
        + media_settings.frame_field_time_interval_ns
        * committed_frame_field_counter);
    };

    /* expected time for all completions of the field, whose completions
       will be polled next, to be ready */
    auto get_finish_time_of_next_field_ns = [&]() { return (
        m_start_send_time_ns
        + media_settings.frame_field_time_interval_ns
        * (completed_frame_field_counter + 1));
    };

    uint64_t scheduled_next_field_start_time_ns = get_send_time_of_next_field_ns();
    uint64_t scheduled_next_field_complete_time_ns = get_finish_time_of_next_field_ns();

    m_marked_token = 0;
    m_handled_token = 0;
    rc = ReturnStatus::success;

    m_commit_ts.reserve(m_app_settings->media.frames_fields_in_mem_block *
                       m_app_settings->media.chunks_in_frame_field);

    LatencyStats tx_delay("Tx latency", m_percentiles);

    size_t chunk_in_field_counter = 0;
    size_t completion_in_field_counter = 0;
    while (likely(rc == ReturnStatus::success && SignalHandler::get_received_signal() < 0)) {
        scheduled_next_field_start_time_ns = static_cast<uint64_t>(get_send_time_of_next_field_ns());
        scheduled_next_field_complete_time_ns = static_cast<uint64_t>(get_finish_time_of_next_field_ns());

        uint64_t ts_now = get_time_now_ns();

        bool is_time_to_fetch_completions = (ts_now >= scheduled_next_field_complete_time_ns);
        bool is_place_to_commit = (committed_frame_field_counter - completed_frame_field_counter) <
                                  media_settings.frames_fields_in_mem_block;

        if (!is_time_to_fetch_completions && !is_place_to_commit) {
            wait_for_next_frame(scheduled_next_field_complete_time_ns);
            continue;
        }

        if (is_time_to_fetch_completions) {
            int fetch_budget = 10;
            do {
                rc = try_process_one_completion(tx_delay);
                if (rc == ReturnStatus::no_completion) {
                    rc = ReturnStatus::success;
                    break;
                }
                if (rc != ReturnStatus::success) {
                    if (rc != ReturnStatus::signal_received) {
                        std::cerr << "Tx completion timeout!" << std::endl;
                    }
                    break;
                }
                completion_in_field_counter++;
                if (--fetch_budget == 0) {
                    break;
                }
            } while (completion_in_field_counter < media_settings.chunks_in_frame_field - LAST_CHUNKS_SKIP_NUM);
            if (completion_in_field_counter == media_settings.chunks_in_frame_field - LAST_CHUNKS_SKIP_NUM) {
                completed_frame_field_counter++;
                completion_in_field_counter = 0;
            }
            if (rc != ReturnStatus::success) {
                break;
            }
        }

        if (is_place_to_commit) {
            do {
                rc = m_send_stream->get_next_chunk(*m_chunk_handler);
                if (unlikely(rc != ReturnStatus::success)) {
                    if (rc != ReturnStatus::signal_received) {
                        std::cerr << "Failed to get a next chunk to send" << std::endl;
                    }
                    break;
                }

                m_send_stream->prepare_chunk_to_send(*m_chunk_handler);
                uint64_t chunk_scheduled_time_ns = static_cast<uint64_t>(
                        scheduled_next_field_start_time_ns +
                        chunk_in_field_counter * m_app_settings->num_of_packets_in_chunk * m_trs);
                m_commit_ts[m_marked_token % m_commit_ts.capacity()] = chunk_scheduled_time_ns;
                uint64_t commit_timestamp_ns = 0;
                if (unlikely(chunk_in_field_counter == 0)) {
                    commit_timestamp_ns = chunk_scheduled_time_ns;
                } else {
                    commit_timestamp_ns = 0;
                }

                bool is_this_chunk_tracked =
                    (chunk_in_field_counter < media_settings.chunks_in_frame_field - LAST_CHUNKS_SKIP_NUM);
                if (is_this_chunk_tracked) {
                    rc = m_chunk_handler->mark_for_tracking(m_marked_token);
                    if (rc != ReturnStatus::success) {
                        std::cerr << "Failed to mark a chunk for tracking" << std::endl;
                        break;
                    }
                }

                bool was_blocked = false;
                do {
                    rc = m_send_stream->commit_chunk(*m_chunk_handler, commit_timestamp_ns);
                    if (unlikely(rc == ReturnStatus::hw_send_queue_full)) {
                        was_blocked = true;
                  }
                } while (unlikely(rc == ReturnStatus::hw_send_queue_full));
                if (unlikely(rc != ReturnStatus::success)) {
                    if (rc != ReturnStatus::signal_received) {
                        std::cerr << "Failed to commit a chunk" << std::endl;
                    }
                    break;
                }
                chunk_in_field_counter++;
                if (is_this_chunk_tracked) {
                    m_marked_token++;
                }
                if (was_blocked) {
                    break;
                }
            } while (likely(rc == ReturnStatus::success &&
                            chunk_in_field_counter < media_settings.chunks_in_frame_field));

            if (chunk_in_field_counter == media_settings.chunks_in_frame_field) {
                committed_frame_field_counter++;
                chunk_in_field_counter = 0;
            }

            if (rc != ReturnStatus::success) {
                std::cerr << "Sending test data failed, aborting test" << std::endl;
                break;
            }

            if (get_time_now_ns() > start_time_ns + m_measure_interval_sec * NS_IN_SEC) {
                break;
            }
        }
    }

    while (m_handled_token != m_marked_token) {
        if (get_time_now_ns() > scheduled_next_field_start_time_ns +
                                media_settings.frame_field_time_interval_ns) {
            std::cerr << "Tx completion timeout!" << std::endl;
            break;
        }

        rc = try_process_one_completion(tx_delay);
        if (rc != ReturnStatus::success && rc != ReturnStatus::no_completion) {
            break;
        }
    }
    if (m_handled_token != m_marked_token) {
        std::cerr << "Tx completion processing error!" << std::endl;
    }

    std::cout << "Tracked " << m_handled_token << " chunks of "
              << m_app_settings->num_of_packets_in_chunk << " packets\n";
    std::cout <<  "Tx completion delay relative to scheduled packet send time.\n";
    tx_delay.calc_percentiles();
    std::cout << "\nAll values are in nanoseconds.\n\n";
    std::cout << tx_delay << std::endl;

    std::this_thread::sleep_for(std::chrono::microseconds(WAIT_SERVER_REPLY_USEC));

    rc = m_receive_stream->get_next_chunk(receive_chunk);
    if (rc != ReturnStatus::success || receive_chunk.get_length() == 0) {
        std::cerr << "No reply from server"<< std::endl;
        return;
    }

    MediaRxLatencyReply reply;
    if (!parse_receive_timing(receive_chunk, reply)) {
        std::cerr << "Invalid server reply" << std::endl;
    } else {
        std::cout <<  "Rx HW timestapm delay relative to scheduled packet send time.\n";
        std::cout << "Avg: " << reply.rx_delay_avg << std::endl;
        std::cout << "Min: " << reply.rx_delay_min << std::endl;
        std::cout << "Max: " << reply.rx_delay_max << std::endl;
    }
}

ReturnStatus MediaTxIONode::try_process_one_completion(LatencyStats &tx_delay)
{
    ReturnStatus rc = m_chunk_handler->poll_for_completion();
    if (rc == ReturnStatus::no_completion) {
        return ReturnStatus::no_completion;
    }
    if (rc != ReturnStatus::success) {
        if (rc != ReturnStatus::signal_received) {
            std::cerr << "Failed polling for Tx completion" << std::endl;
        }
        return rc;
    }
    uint64_t tx_hw_timestamp;
    uint64_t compl_token;
    rc = m_chunk_handler->get_last_completion_info(tx_hw_timestamp, compl_token);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to get completion info for a sent chunk" << std::endl;
        return rc;
    }
    if (compl_token != m_handled_token) {
        std::cerr << "Out-of-order Tx completion (expected " << m_handled_token << " got "
                  << compl_token << ")" << std::endl;
        return ReturnStatus::failure;
    }
    int64_t delta = tx_hw_timestamp - m_commit_ts[m_handled_token % m_commit_ts.capacity()] -
                    static_cast<uint64_t>((m_app_settings->num_of_packets_in_chunk - 1 )* m_trs);
    tx_delay.update(delta);
    m_handled_token++;
    return ReturnStatus::success;
}

bool MediaTxIONode::parse_receive_timing(ReceiveChunk& chunk, MediaRxLatencyReply& timing)
{
    ReceivePacketInfo packet_info = chunk.get_packet_info(0);
    size_t packet_size = packet_info.get_packet_sub_block_size(0);
    if (packet_size != sizeof(timing)) {
        std::cerr << "Invalid reply size " << packet_size << " (" << sizeof(timing) << ")"
                  << std::endl;
        return false;
    }
    const void *payload =
        static_cast<const rmx_input_completion_metadata *>(chunk.get_payload_ptr());
    m_header_mem_utils->memory_copy(&timing, payload, sizeof(timing));
    return true;
}

MediaRxIONode::MediaRxIONode(
        const LatencyNodeSettings& settings,
        std::shared_ptr<MemoryUtils> header_mem_utils,
        std::shared_ptr<MemoryUtils> payload_mem_utils,
        time_handler_ns_cb_t get_time_ns_cb
    ) : GenericLatencyIONode(
                settings,
                StreamDimensions(DEFAULT_NUM_OF_SEND_CHUNKS, 1, 0, MEDIA_TX_REPLY_SIZE),
                StreamDimensions(DEFAULT_NUM_OF_RECEIVE_CHUNKS, 1, 0, DEFAULT_RESPONSE_SIZE),
                std::move(header_mem_utils), std::move(payload_mem_utils), std::move(get_time_ns_cb)),
    m_app_settings(settings.app)
{
    m_app_settings->num_of_total_streams = 1;
    m_app_settings->media.frames_fields_in_mem_block = 1;
    auto rc = initialize_media_settings(*m_app_settings);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to initialize media settings" << std::endl;
        throw std::runtime_error("Failed to initialize media settings");
    }
    m_receive_dim.header_size = m_app_settings->packet_app_header_size;
    m_receive_dim.payload_size = m_app_settings->packet_payload_size;
    m_receive_dim.num_of_chunks = m_app_settings->num_of_chunks_in_mem_block;
    m_receive_dim.num_of_packets_in_chunk = m_app_settings->num_of_packets_in_chunk;
}

void MediaRxIONode::compose_reply(std::shared_ptr<GenericChunk> commit_chunk, const LatencyStats& stats,
                       uint64_t dropped_pkt_cnt)
{
    MediaRxLatencyReply reply;
    reply.rx_delay_min = stats.get_min();
    reply.rx_delay_avg = stats.get_avg();
    reply.rx_delay_max = stats.get_max();
    reply.packets_lost = dropped_pkt_cnt;
    auto& pkt = commit_chunk->get_packet(0);
    auto& mreg = pkt[0];
    m_header_mem_utils->memory_copy(mreg.addr, &reply, sizeof(reply));
}

ReturnStatus MediaRxIONode::prepare_reply_chunk(std::shared_ptr<GenericChunk>& chunk)
{
    ReturnStatus rc;
    rc = m_send_stream->get_next_chunk(chunk);
    if (rc == ReturnStatus::no_free_chunks) {
        std::cerr << "Error, no free chunk to send..." << std::endl;
        return rc;
    }
    if (rc != ReturnStatus::success) {
        if (rc != ReturnStatus::signal_received) {
            std::cerr << "Error getting a chunk to send..." << std::endl;
        }
        return rc;
    }

    auto& pkt = chunk->get_packet(0);
    pkt[0].length = sizeof(MediaRxLatencyReply);
    rc = chunk->apply_packets_layout();
    if (rc != ReturnStatus::success) {
        if (rc != ReturnStatus::signal_received) {
            std::cerr << "Error preparing a chunk to send..." << std::endl;
        }
    }
    return rc;
}

ReturnStatus MediaRxIONode::get_rtp_param(const uint8_t* rtp_hdr, uint32_t& sequence_number,
                                          uint32_t& timestamp, bool& m_bit, bool& f_bit)
{
    timestamp = 0;

    if (((rtp_hdr[0] & 0xC0) != 0x80)) {
        return ReturnStatus::failure;
    }

    uint8_t cc = 0x0F & rtp_hdr[0];
    uint8_t offset = cc * RTP_HEADER_CSRC_GRANULARITY_BYTES;

    sequence_number = rtp_hdr[3] | rtp_hdr[2] << 8;
    sequence_number |= rtp_hdr[offset + 12] << 24 | rtp_hdr[offset + 13] << 16;
    f_bit = !!(rtp_hdr[offset + 16] & 0x80);

    timestamp = ntohl(*(uint32_t *) (((uint8_t*) rtp_hdr) + 4));
    m_bit = !!(rtp_hdr[1] & 0x80);
    return ReturnStatus::success;
}

uint64_t MediaRxIONode::calc_next_frame_start(uint64_t last_pkt_ts)
{
    double send_time_ns = static_cast<double>(last_pkt_ts);
    double t_frame_ns;

    if (m_app_settings->media.video_scan_type == VideoScanType::Progressive) {
        t_frame_ns = m_app_settings->media.frame_field_time_interval_ns;
    } else {
        t_frame_ns = m_app_settings->media.frame_field_time_interval_ns * 2;
    }

    uint64_t N = static_cast<uint64_t>(send_time_ns / t_frame_ns) + 1;
    return N * t_frame_ns;
}

void MediaRxIONode::receive_send()
{
    ReturnStatus rc = ReturnStatus::success;
    std::shared_ptr<GenericChunk> commit_chunk;
    ReceiveChunk receive_chunk(m_receive_stream->get_id(), is_receive_hds());
    m_receive_stream->set_completion_moderation(0, MAX_RX_CHUNK_SIZE, 0);
    LatencyStats rx_latency("Rx latency", m_percentiles);
    RxState rx_state = RxState::syncing;
    uint64_t frame_start_ts = 0;
    uint32_t frame_first_seq_num = 0;
    uint32_t last_seq_num = 0;
    uint64_t last_packet_ts = get_time_now_ns();
    uint64_t dropped_pkt_cnt = 0;
    uint64_t measure_cnt = 0;
    double tro;
    double trs;
    calculate_tro_trs(m_app_settings->media, tro, trs);

    while (rc != ReturnStatus::failure && rc != ReturnStatus::signal_received &&
                  SignalHandler::get_received_signal() < 0) {

        rc = m_receive_stream->get_next_chunk(receive_chunk);
        if (rc != ReturnStatus::success) {
            if (rc == ReturnStatus::signal_received) {
                if (rx_latency.get_cnt())
                {
                    std::cout << "\nAll values are in nanoseconds.\n\n";
                    rx_latency.calc_percentiles();
                    std::cout << rx_latency << std::endl;
                }
            } else {
                std::cerr << "Failed to get data chunk from Rx stream" << std::endl;
            }
            return;
        }

        if (receive_chunk.get_length() == 0) {
            if (rx_state == RxState::syncing) {
                continue;
            } else if (rx_state == RxState::receiving) {
                if (get_time_now_ns() - last_packet_ts >= WAIT_SERVER_RECEIVE_NSEC) {
                    rx_state = RxState::paused;
                }
                continue;
            } else {
                auto now = get_time_now_ns();
                if (now - last_packet_ts <= WAIT_SERVER_INACT_NSEC) {
                    continue;
                }
                std::cout << "Receive stream stopped, received "
                          << measure_cnt << " packets" << std::endl;
                measure_cnt = 0;
                if (rx_latency.get_cnt()) {
                    std::cout << "\nAll values are in nanoseconds.\n\n";
                    rx_latency.calc_percentiles();
                    std::cout << rx_latency << std::endl;
                    rc = prepare_reply_chunk(commit_chunk);
                    if (rc != ReturnStatus::success) {
                        break;
                    }
                    compose_reply(commit_chunk, rx_latency, dropped_pkt_cnt);
                    rx_latency.reset();
                    rc = m_send_stream->commit_chunk(commit_chunk, 0);
                    if (rc != ReturnStatus::success) {
                        std::cerr << "Failed to send reply to client" << std::endl;
                        break;
                    };
                }
                rx_state = RxState::syncing;
                dropped_pkt_cnt = 0;
                continue;
            }
        }

        const uint8_t* data_ptr;
        size_t stride_size;
        if (!is_receive_hds()) {
            data_ptr = static_cast<const uint8_t*>(receive_chunk.get_payload_ptr());
            stride_size = m_receive_stream->get_payload_stride_size();
        } else {
            data_ptr = static_cast<const uint8_t*>(receive_chunk.get_header_ptr());
            stride_size = m_receive_stream->get_header_stride_size();
        }
        auto strides_cnt = receive_chunk.get_length();
        for (size_t pkt_idx_in_chunk = 0; pkt_idx_in_chunk < strides_cnt; pkt_idx_in_chunk++) {
            uint32_t sequence_number;
            uint32_t rtp_timestamp;
            bool bit_m;
            bool bit_f;
            auto packet_info = receive_chunk.get_packet_info(pkt_idx_in_chunk);
            last_packet_ts = packet_info.get_packet_timestamp();

            const uint8_t* pkt_ptr = data_ptr + pkt_idx_in_chunk * stride_size;
            if (get_rtp_param(pkt_ptr, sequence_number, rtp_timestamp, bit_m, bit_f) !=
                ReturnStatus::success) {
                continue;
            }

            if (rx_state == RxState::receiving) {
                if (sequence_number != last_seq_num + 1) {
                    dropped_pkt_cnt += sequence_number - (last_seq_num + 1);
                }
                uint32_t pkt_idx_in_frame = sequence_number - frame_first_seq_num;
                if (pkt_idx_in_frame >= m_app_settings->media.packets_in_frame_field) {
                    std::cerr << "Invalid packet index in frame " << pkt_idx_in_frame << std::endl;
                    rx_latency.reset();
                    rx_state = RxState::syncing;
                    continue;
                }
                measure_cnt++;
                int64_t delta = last_packet_ts -
                                static_cast<int64_t>(frame_start_ts + tro + trs * pkt_idx_in_frame);

                rx_latency.update(delta);
                last_seq_num = sequence_number;
            }

            if (bit_m) {
                frame_start_ts = calc_next_frame_start(last_packet_ts);
                if (rx_state == RxState::syncing) {
                    std::cout << "Receive stream synced" << std::endl;
                }
                rx_state = RxState::receiving;
                last_seq_num = sequence_number;
                frame_first_seq_num = sequence_number + 1;
            }
        }
    }
}
