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

#include <thread>
#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>
#include <cstring>
#include <chrono>

#include <rivermax_api.h>
#include "rt_threads.h"
#include <api/rmax_apps_lib_api.h>
#include "generic_latency_io_node.h"

using namespace ral::io_node;
using namespace ral::lib::core;
using namespace ral::lib::services;

GenericLatencyIONode::GenericLatencyIONode(
        const LatencyNodeSettings& settings,
        const StreamDimensions& send_dim,
        const StreamDimensions& receive_dim,
        std::shared_ptr<MemoryUtils> header_mem_utils,
        std::shared_ptr<MemoryUtils> payload_mem_utils,
        time_handler_ns_cb_t time_handler_cb
    ) : LatencyIONode(settings, header_mem_utils, payload_mem_utils, time_handler_cb),
    m_send_dim(send_dim),
    m_receive_dim(receive_dim),
    m_hw_queue_full_sleep_us(settings.app->hw_queue_full_sleep_us),
    m_track_completions(settings.track_completions),
    m_buffer_writer(std::unique_ptr<IBufferWriter>(dynamic_cast<IBufferWriter*>(new GenericBufferWriter())))
{
}

std::ostream& GenericLatencyIONode::print(std::ostream& out) const
{
    out << "+#############################################\n"
        << "| Thread ID: 0x" << std::hex << std::this_thread::get_id() << std::dec << "\n"
        << "| CPU core affinity: " << m_cpu_core_affinity << "\n"
        << "| Tx Header address: " << m_send_header_region.addr << "\n"
        << "| Tx Header length: " << m_send_header_region.length << "[B]" << "\n"
        << "| Tx Header key: " << m_send_header_region.mkey << "\n"
        << "| Tx Payload address: " << m_send_payload_region.addr << "\n"
        << "| Tx Payload length: " << m_send_payload_region.length << "[B]" << "\n"
        << "| Tx Payload key: " << m_send_payload_region.mkey << "\n"
        << "| Rx Header address: " << m_receive_header_region.addr << "\n"
        << "| Rx Header length: " << m_receive_header_region.length << "[B]" << "\n"
        << "| Rx Header key: " << m_receive_header_region.mkey << "\n"
        << "| Rx Payload address: " << m_receive_payload_region.addr << "\n"
        << "| Rx Payload length: " << m_receive_payload_region.length << "[B]" << "\n"
        << "| Rx Payload key: " << m_receive_payload_region.mkey << "\n"
        << "+#############################################\n";
    return out;
}

void GenericLatencyIONode::initialize_send_stream()
{
    if (m_gpu_direct_tx && (m_send_dim.header_size == 0)) {
        m_send_dim.header_size = RTP_HEADER_SIZE;
        m_send_dim.payload_size -= RTP_HEADER_SIZE;
    }

    GenericStreamSettings settings(m_network_address,
            true,
            pp_rate_t{0, 0},
            m_send_dim.num_of_chunks,
            m_send_dim.num_of_packets_in_chunk,
            static_cast<uint16_t>(m_send_dim.payload_size),
            static_cast<uint16_t>(m_send_dim.header_size));

        m_send_stream = std::shared_ptr<GenericSendStream>(new GenericSendStream(settings));
}

ReturnStatus GenericLatencyIONode::query_memory_size(size_t& tx_header_size, size_t& tx_payload_size,
                                                     size_t& rx_header_size, size_t& rx_payload_size)
{
    if (m_receive_stream->query_buffer_size(rx_header_size, rx_payload_size) != ReturnStatus::success) {
        return ReturnStatus::failure;
    }

    tx_header_size = m_send_stream->get_header_memory_length();
    tx_payload_size = m_send_stream->get_payload_memory_length();
    return ReturnStatus::success;
};

void GenericLatencyIONode::distribute_memory_for_streams(rmx_mem_region& tx_header_mreg,
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
}

void GenericLatencyIONode::print_parameters()
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

ReturnStatus GenericLatencyIONode::create_send_stream()
{
    ReturnStatus rc;

    rc = m_send_stream->create_stream();

    if (rc == ReturnStatus::success) {
        if (is_send_hds()) {
            m_send_stream->initialize_chunks(m_send_header_region, m_send_payload_region);
        } else {
            m_send_stream->initialize_chunks(m_send_payload_region);
        }
    }

    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to create send stream" << std::endl;
        return rc;
    }

    return ReturnStatus::success;
}

ReturnStatus GenericLatencyIONode::destroy_send_stream()
{
    ReturnStatus rc;

    rc = m_send_stream->destroy_stream();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to destroy send stream" << std::endl;
        return rc;
    }

    return ReturnStatus::success;
}

void GenericLatencyIONode::initialize_receive_stream(const TwoTupleFlow& flow)
{
    if (m_gpu_direct_rx  && (m_receive_dim.header_size == 0)) {
        m_receive_dim.header_size = RTP_HEADER_SIZE;
        m_receive_dim.payload_size -= RTP_HEADER_SIZE;
    }

    ReceiveStreamSettings settings(flow,
        RMX_INPUT_APP_PROTOCOL_PACKET,
        RMX_INPUT_TIMESTAMP_SYNCED,
        {RMX_INPUT_STREAM_CREATE_INFO_PER_PACKET},
        m_receive_dim.num_of_chunks * m_receive_dim.num_of_packets_in_chunk,
        m_receive_dim.payload_size,
        m_receive_dim.header_size);

    std::cout << "Init receive stream with " << m_receive_dim.num_of_chunks *
                                                m_receive_dim.num_of_packets_in_chunk
              << " packets of "
              << (is_receive_hds() ?
                    std::to_string(m_receive_dim.header_size) + "+" : "")
              << m_receive_dim.payload_size << " bytes" << std::endl;
    m_receive_stream = std::shared_ptr<ReceiveStream>(new ReceiveStream(settings));
}

ReturnStatus GenericLatencyIONode::create_receive_stream()
{
    ReturnStatus rc = m_receive_stream->create_stream();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to create response stream" << std::endl;
        return rc;
    }
    return ReturnStatus::success;
}

ReturnStatus GenericLatencyIONode::attach_receive_flow()
{
    ReturnStatus rc = m_receive_stream->attach_flow(m_receive_flow);
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to attach flow to response stream" << std::endl;
        return rc;
    }

    return ReturnStatus::success;
}

ReturnStatus GenericLatencyIONode::detach_receive_flow()
{
    ReturnStatus rc = m_receive_stream->detach_flow(m_receive_flow);
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to detach flow from response stream" << std::endl;
        return rc;
    }

    return ReturnStatus::success;
}

ReturnStatus GenericLatencyIONode::destroy_receive_stream()
{
    ReturnStatus rc = m_receive_stream->destroy_stream();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to create response stream" << std::endl;
        return rc;
    }
    return ReturnStatus::success;
}

PingPongIONode::PingPongIONode(
        const LatencyNodeSettings& settings,
        std::shared_ptr<MemoryUtils> header_mem_utils,
        std::shared_ptr<MemoryUtils> payload_mem_utils,
        time_handler_ns_cb_t time_handler_cb
    ) : GenericLatencyIONode(settings,
                      StreamDimensions(DEFAULT_NUM_OF_SEND_CHUNKS, 1,
                      0, settings.app->packet_payload_size),
                      StreamDimensions(DEFAULT_NUM_OF_RECEIVE_CHUNKS, 1,
                      0, settings.app->packet_payload_size),
                      header_mem_utils, payload_mem_utils, time_handler_cb)
{
}

void PingPongIONode::prepare_send_buffer()
{
    for (size_t chunk_index = 0; chunk_index < m_send_stream->get_num_of_chunks(); chunk_index++) {
        auto chunk = m_send_stream->get_chunk(chunk_index);
        for (size_t packet_index = 0; packet_index < chunk->get_length(); packet_index++) {
            auto& packet = chunk->get_packet(packet_index);
            m_header_mem_utils->memory_set(reinterpret_cast<void*>(packet[0].addr),
                                           '0' + (packet_index % 9), packet[0].length);
            if (packet.size() > 1) {
                m_payload_mem_utils->memory_set(reinterpret_cast<void*>(packet[1].addr),
                                                '0' + (packet_index % 9), packet[1].length);
            }
        }
    }
}

void PingPongIONode::send_receive()
{
    ReturnStatus rc = ReturnStatus::success;
    std::shared_ptr<GenericChunk> commit_chunk;
    ReceiveChunk receive_chunk(m_receive_stream->get_id(), false);
    m_receive_stream->set_completion_moderation(1, 1, 1000000);

    size_t skip_packets = DEFAULT_SKIP_PACKETS;
    size_t packet_cnt = 0;
    uint64_t start_time_ns = get_time_now_ns();
    uint64_t send_ts = 0, receive_ts = 0;
    uint64_t token = 0;
    uint64_t tx_hw_timestamp = 0;
    uint64_t rx_hw_timestamp = 0;
    LatencyStats end_to_end("End-to-End Latency", m_percentiles);
    LatencyStats tx_latency("Tx Latency", m_percentiles);
    LatencyStats rx_latency("Rx Latency", m_percentiles);

    while (rc != ReturnStatus::failure && rc != ReturnStatus::signal_received &&
                  SignalHandler::get_received_signal() < 0) {
        for (size_t chunk = 0; chunk < 1; chunk++) {
            rc = m_send_stream->get_next_chunk(commit_chunk);
            if (rc == ReturnStatus::no_free_chunks) {
                continue;
            }
            if (m_track_completions) {
                rc = commit_chunk->mark_for_tracking(token++);
                if (rc != ReturnStatus::success) {
                    std::cerr << "Failed to mark a chunk for tracking" << std::endl;
                    break;
                }
            }
            rc = commit_chunk->apply_packets_layout();
            if (rc != ReturnStatus::success) {
                break;
            }
            send_ts = get_time_now_ns();
            do {
                rc = m_send_stream->commit_chunk(commit_chunk, 0);
                switch (rc) {
                    case ReturnStatus::hw_send_queue_full:
                        std::this_thread::sleep_for(std::chrono::microseconds(m_hw_queue_full_sleep_us));
                        break;
                    case ReturnStatus::failure:
                        std::cerr << "Failed to send a chunk" << std::endl;
                        break;
                    case ReturnStatus::signal_received:
                        std::cerr << "Received signal when sending a chunk" << std::endl;
                        break;
                    default:
                        break;
                }
            } while (rc == ReturnStatus::hw_send_queue_full);
        }
        if (rc != ReturnStatus::success) {
            break;
        }

        rc = m_receive_stream->get_next_chunk(receive_chunk);
        receive_ts = get_time_now_ns();

        if (rc != ReturnStatus::success) {
            if (rc != ReturnStatus::signal_received) {
                std::cerr << "Failed to get data chunk from stream"<< std::endl;
            }
            return;
        }

        if (m_track_completions) {
            rc = commit_chunk->poll_for_completion();
            if (rc != ReturnStatus::success) {
                std::cerr << "Failed to get a completion for a sent chunk" << std::endl;
                break;
            }
            uint64_t compl_token;
            rc = commit_chunk->get_last_completion_info(tx_hw_timestamp, compl_token);
            if (rc != ReturnStatus::success) {
                std::cerr << "Failed to get completion info for a sent chunk" << std::endl;
                break;
            }
            if (compl_token + 1 != token) {
                std::cerr << "Out-of-order Tx completion" << std::endl;
                break;
            }
            rx_hw_timestamp =
                    rmx_input_get_completion_timestamp_first(receive_chunk.get_completion());
        }

        if (skip_packets) {
            skip_packets--;
            continue;
        }

        if (receive_chunk.get_length() != 0) {
            int64_t delta = (receive_ts - send_ts) / 2;
            end_to_end.update(delta);
            packet_cnt++;

            if (m_track_completions) {
                int64_t tx_delta = tx_hw_timestamp - send_ts;
                tx_latency.update(tx_delta);

                int64_t rx_delta = receive_ts - rx_hw_timestamp;
                rx_latency.update(rx_delta);
            }
        } else {
            std::cerr << "-";
        }

        if (std::chrono::nanoseconds(receive_ts) > std::chrono::nanoseconds(start_time_ns) +
                                                   std::chrono::seconds(m_measure_interval_sec)) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::microseconds(m_sleep_between_operations_us));
    }

    std::cout << "Total replies received: " << packet_cnt << std::endl;
    if (packet_cnt) {
        end_to_end.calc_percentiles();
        std::cout << "\nAll values are in nanoseconds.\n\n";
        std::cout << end_to_end;
        if (m_track_completions) {
            tx_latency.calc_percentiles();
            std::cout << tx_latency;
            rx_latency.calc_percentiles();
            std::cout << rx_latency;
        }
    }
}

void PingPongIONode::receive_send()
{
    ReturnStatus rc = ReturnStatus::success;
    std::shared_ptr<GenericChunk> commit_chunk;
    ReceiveChunk receive_chunk(m_receive_stream->get_id(), false);
    m_receive_stream->set_completion_moderation(1, 1, 0);

    while (rc != ReturnStatus::failure && rc != ReturnStatus::signal_received &&
                  SignalHandler::get_received_signal() < 0) {

        /* Prepare a reply chunk in advance */
        rc = m_send_stream->get_next_chunk(commit_chunk);
        if (rc == ReturnStatus::no_free_chunks) {
            std::cerr << "Error, no free chunk to send..." << std::endl;
            continue;
        }
        if (rc != ReturnStatus::success) {
            if (rc != ReturnStatus::signal_received) {
                std::cerr << "Error getting a chunk to send..." << std::endl;
            }
            break;
        }
        rc = commit_chunk->apply_packets_layout();
        if (rc != ReturnStatus::success) {
            if (rc != ReturnStatus::signal_received) {
                std::cerr << "Error preparing a chunk to send..." << std::endl;
            }
            break;
        }

        rc = m_receive_stream->get_next_chunk(receive_chunk);
        if (rc != ReturnStatus::success) {
            if (rc != ReturnStatus::signal_received) {
                std::cerr << "Failed to get data chunk from Rx stream" << std::endl;
            }
            return;
        }

        if (receive_chunk.get_length() == 0) {
            continue;
        }

        do {
            rc = m_send_stream->commit_chunk(commit_chunk, 0);
            switch (rc) {
                case ReturnStatus::hw_send_queue_full:
                    std::this_thread::sleep_for(std::chrono::microseconds(m_hw_queue_full_sleep_us));
                    break;
                case ReturnStatus::failure:
                    std::cerr << "Failed to send chunk of Tx stream" << std::endl;
                    break;
                case ReturnStatus::signal_received:
                    std::cerr << "Received signal when send chunk of Tx stream" << std::endl;
                    break;
                default:
                    break;
            }
        } while (rc == ReturnStatus::hw_send_queue_full);
    }
}

FrameIONode::FrameIONode(
        const LatencyNodeSettings& settings,
        std::shared_ptr<MemoryUtils> header_mem_utils,
        std::shared_ptr<MemoryUtils> payload_mem_utils,
        time_handler_ns_cb_t time_handler_cb
    ) : GenericLatencyIONode(
                settings,
                settings.client_mode ?
                        StreamDimensions(settings.app->num_of_chunks,
                                         settings.app->num_of_packets_in_chunk,
                                         settings.app->packet_app_header_size,
                                         settings.app->packet_payload_size) :
                        StreamDimensions(DEFAULT_NUM_OF_SEND_CHUNKS, 1,
                                         0, DEFAULT_RESPONSE_SIZE),
                settings.client_mode ?
                        StreamDimensions(DEFAULT_NUM_OF_RECEIVE_CHUNKS, 1,
                                         0, DEFAULT_RESPONSE_SIZE) :
                        StreamDimensions(settings.app->num_of_chunks,
                                         settings.app->num_of_packets_in_chunk,
                                         settings.app->packet_app_header_size,
                                         settings.app->packet_payload_size),
                header_mem_utils, payload_mem_utils, time_handler_cb),
    m_rx_next_pkt_num{0},
    m_rx_iteration{0},
    m_rx_drop_cnt{0}
{
}

void FrameIONode::prepare_send_buffer()
{
    for (size_t chunk_index = 0; chunk_index < m_send_stream->get_num_of_chunks(); chunk_index++) {
        auto chunk = m_send_stream->get_chunk(chunk_index);
        for (size_t packet_index = 0; packet_index < chunk->get_length(); packet_index++) {
            auto& packet = chunk->get_packet(packet_index);
            m_header_mem_utils->memory_set(reinterpret_cast<void*>(packet[0].addr),
                                           '0' + (packet_index % 9), packet[0].length);
            if (packet.size() > 1) {
                m_payload_mem_utils->memory_set(reinterpret_cast<void*>(packet[1].addr),
                                                '0' + (packet_index % 9), packet[1].length);
            }
        }
    }
}

void FrameIONode::prepare_chunks(uint32_t iteration)
{
    uint32_t pkt_id = 0;
    for (size_t chunk_index = 0; chunk_index < m_send_stream->get_num_of_chunks(); chunk_index++) {
        auto chunk = m_send_stream->get_chunk(chunk_index);
        for (size_t pkt_num = 0; pkt_num < chunk->get_length(); pkt_num++) {
            packet_header hdr {iteration, pkt_id++};
            auto mreg = chunk->get_packet(pkt_num)[0];
            m_header_mem_utils->memory_copy(mreg.addr, &hdr, sizeof(hdr));
        }
    }
}

bool FrameIONode::parse_receive_timing(ReceiveChunk& chunk, receive_timing& timing)
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

void FrameIONode::send_receive()
{
    ReturnStatus rc = ReturnStatus::success;
    std::shared_ptr<GenericChunk> commit_chunk;
    ReceiveChunk receive_chunk(m_receive_stream->get_id(), false);
    m_receive_stream->set_completion_moderation(1, 1, 1000000);

    size_t skip_packets = DEFAULT_SKIP_PACKETS;
    size_t packet_cnt = 0;
    uint64_t start_time_ns = get_time_now_ns();
    uint64_t send_ts = 0, receive_ts = 0;
    uint64_t token = 0;
    uint64_t tx_hw_timestamp = 0;
    LatencyStats total_time("Total SW Time (Commit to Reply)", m_percentiles);
    LatencyStats commit_to_tx_done("Tx Commit to Tx Done TS", m_percentiles);
    LatencyStats commit_to_rx_ts("Tx Commit to Reply Rx TS", m_percentiles);
    LatencyStats commit_to_rx_first("Tx Commit to First Pkt Rx TS", m_percentiles);
    LatencyStats commit_to_rx_last("Tx Commit to Last Pkt Rx TS", m_percentiles);
    LatencyStats commit_to_rx_soft("Tx Commit to Rx SW done", m_percentiles);
    LatencyStats rx_first_hw_to_soft("Rx First HW TS to Rx SW done", m_percentiles);
    LatencyStats soft_feedback("Rx SW done to Reply", m_percentiles);
    uint64_t rx_hw_timestamp = 0;

    uint32_t iteration = 0;

    std::this_thread::sleep_for(std::chrono::microseconds(100000));

    while (rc == ReturnStatus::success && SignalHandler::get_received_signal() < 0) {

        std::this_thread::sleep_for(std::chrono::microseconds(m_sleep_between_operations_us));
        prepare_chunks(iteration);

        send_ts = get_time_now_ns();
        for (size_t chunk = 0; chunk < m_send_dim.num_of_chunks; chunk++) {
            if (rc != ReturnStatus::success) {
                break;
            }

            do {
                rc = m_send_stream->blocking_get_next_chunk(commit_chunk, BLOCKING_CHUNK_RETRIES);
            } while (unlikely(rc == ReturnStatus::no_free_chunks));

            if (rc != ReturnStatus::success) {
                if (rc != ReturnStatus::signal_received) {
                    std::cerr << "Failed to get a next chunk to send" << std::endl;
                }
                break;
            }
            if (m_track_completions) {
                if (chunk + 1 == m_send_dim.num_of_chunks) {
                    rc = commit_chunk->mark_for_tracking(token++);
                    if (rc != ReturnStatus::success) {
                        std::cerr << "Failed to mark a chunk for tracking" << std::endl;
                        break;
                    }
                }
            }
            rc = commit_chunk->apply_packets_layout();
            if (rc != ReturnStatus::success) {
                if (rc != ReturnStatus::signal_received) {
                    std::cerr << "Cannot apply packet layout" << std::endl;
                }
                break;
            }
            do {
                rc = m_send_stream->commit_chunk(commit_chunk, 0);
                switch (rc) {
                    case ReturnStatus::success:
                        break;
                    case ReturnStatus::hw_send_queue_full:
                        std::this_thread::sleep_for(
                                std::chrono::microseconds(m_hw_queue_full_sleep_us));
                        break;
                    case ReturnStatus::failure:
                        std::cerr << "Failed to send a chunk" << std::endl;
                        break;
                    case ReturnStatus::signal_received:
                        std::cerr << "Received signal when sending a chunk" << std::endl;
                        break;
                    default:
                        std::cerr << "Something strange happened ("
                                  << static_cast<std::underlying_type<ReturnStatus>::type>(rc)
                                  << ")" << std::endl;
                        break;
                }
            } while (rc == ReturnStatus::hw_send_queue_full);
        }
        if (rc != ReturnStatus::success) {
            break;
        }

        iteration++;

        rc = m_receive_stream->get_next_chunk(receive_chunk);
        receive_ts = get_time_now_ns();

        if (rc != ReturnStatus::success) {
            if (rc != ReturnStatus::signal_received) {
                std::cerr << "Failed to get data chunk from stream"<< std::endl;
            }
            break;
        }

        if (m_track_completions) {
            /* Need to poll multiple times because only the last completion is marked for tracking,
               and each poll can process only a limited number of HW completions */
            for (size_t i = 0; i < m_send_dim.num_of_chunks; i++) {
                rc = commit_chunk->poll_for_completion();
                if (rc == ReturnStatus::success) {
                    break;
                }
            }
            if (rc != ReturnStatus::success) {
                std::cerr << "Failed to get a completion for the last sent chunk" << std::endl;
                break;
            }
            uint64_t compl_token;
            rc = commit_chunk->get_last_completion_info(tx_hw_timestamp, compl_token);
            if (rc != ReturnStatus::success) {
                std::cerr << "Failed to get completion info for a sent chunk" << std::endl;
                break;
            }
            if (compl_token + 1 != token) {
                std::cerr << "Out-of-order Tx completion (expected " << token - 1 << " got "
                          << compl_token << ")" << std::endl;
                break;
            }
            if (receive_chunk.get_length() != 0) {
                rx_hw_timestamp =
                        rmx_input_get_completion_timestamp_first(receive_chunk.get_completion());
            }
        }

        if (std::chrono::nanoseconds(receive_ts) > std::chrono::nanoseconds(start_time_ns) +
                                                   std::chrono::seconds(m_measure_interval_sec)) {
            break;
        }

        if (skip_packets) {
            skip_packets--;
            continue;
        }

        if (receive_chunk.get_length() == 0) {
            std::cerr << "No reply" << std::endl;
            continue;
        }

        receive_timing rx_timing;
        if (parse_receive_timing(receive_chunk, rx_timing)) {
            if (rx_timing.iter_num + 1 != iteration) {
                std::cerr << "Reply ordering error" << std::endl;
                break;
            }
            if (!rx_timing.valid) {
                std::cerr << "Rx data loss detected" << std::endl;
                continue;
            }
        } else {
            std::cerr << "Broken reply" << std::endl;
            break;
        }

        int64_t delta = (receive_ts - send_ts);
        total_time.update(delta);
        packet_cnt++;

        if (m_track_completions) {
            int64_t tx_delta = tx_hw_timestamp - send_ts;
            commit_to_tx_done.update(tx_delta);
            int64_t rx_delta = rx_hw_timestamp - send_ts;
            commit_to_rx_ts.update(rx_delta);
        }

        int64_t rx_delta_first = rx_timing.first_rx_ts - send_ts;
        commit_to_rx_first.update(rx_delta_first);
        int64_t rx_delta_last = rx_timing.last_rx_ts - send_ts;
        commit_to_rx_last.update(rx_delta_last);
        int64_t rx_delta_soft = rx_timing.soft_rx_ts - send_ts;
        commit_to_rx_soft.update(rx_delta_soft);
        int64_t rx_delta_hw_to_sw = rx_timing.soft_rx_ts - rx_timing.first_rx_ts;
        rx_first_hw_to_soft.update(rx_delta_hw_to_sw);
        int64_t soft_feedback_delta = receive_ts - rx_timing.soft_rx_ts;
        soft_feedback.update(soft_feedback_delta);

    }

    std::cout << "Total replies received: " << packet_cnt << std::endl;
    if (packet_cnt) {
        total_time.calc_percentiles();
        std::cout << "\nAll values are in nanoseconds.\n\n";
        std::cout << total_time;
        if (m_track_completions) {
            commit_to_tx_done.calc_percentiles();
            std::cout << commit_to_tx_done;
            commit_to_rx_ts.calc_percentiles();
            std::cout << commit_to_rx_ts;
        }
        commit_to_rx_first.calc_percentiles();
        std::cout << commit_to_rx_first;
        commit_to_rx_last.calc_percentiles();
        std::cout << commit_to_rx_last;
        commit_to_rx_soft.calc_percentiles();
        std::cout << commit_to_rx_soft;
        rx_first_hw_to_soft.calc_percentiles();
        std::cout << rx_first_hw_to_soft;
        soft_feedback.calc_percentiles();
        std::cout << soft_feedback;
    }
}

void FrameIONode::parse_packet_headers(ReceiveChunk& chunk, bool& has_last_packet,
                                       bool& valid)
{
    const uint8_t* header_ptr;
    size_t stride_size;
    if (is_receive_hds()) {
        header_ptr = static_cast<const uint8_t*>(chunk.get_header_ptr());
        stride_size = m_receive_stream->get_header_stride_size();
    } else {
        header_ptr = static_cast<const uint8_t*>(chunk.get_payload_ptr());
        stride_size = m_receive_stream->get_payload_stride_size();
    }
    auto strides_cnt = chunk.get_length();
    auto packets_in_frame = m_receive_dim.num_of_chunks * m_receive_dim.num_of_packets_in_chunk;
    valid = true;
    for (size_t pkt_idx = 0; pkt_idx < strides_cnt; pkt_idx++) {
        auto header = reinterpret_cast<const packet_header*>(header_ptr);
        if (header->pkt_num != m_rx_next_pkt_num) {
            valid = false;
            if (header->pkt_num > m_rx_next_pkt_num) {
                m_rx_drop_cnt += static_cast<size_t>(header->pkt_num - m_rx_next_pkt_num);
            } else {
                m_rx_drop_cnt += static_cast<size_t>((packets_in_frame + header->pkt_num) -
                                                      m_rx_next_pkt_num);
            }
        }
        m_rx_next_pkt_num = header->pkt_num + 1;
        if (header->pkt_num == packets_in_frame - 1) {
            has_last_packet = true;
            m_rx_iteration = header->iter_num;
            m_rx_next_pkt_num = 0;
            valid = valid && (pkt_idx == strides_cnt - 1);
        }
        header_ptr += stride_size;
    }
}

void FrameIONode::compose_reply(std::shared_ptr<GenericChunk> commit_chunk,
                                uint64_t first_packet_ts, uint64_t last_packet_ts,
                                uint64_t receive_ts, uint32_t iteration, bool valid)
{
    receive_timing timing {first_packet_ts, last_packet_ts, receive_ts, iteration, valid};
    auto& pkt = commit_chunk->get_packet(0);
    auto& mreg = pkt[0];
    m_header_mem_utils->memory_copy(mreg.addr, &timing, sizeof(timing));
}

ReturnStatus FrameIONode::prepare_reply_chunk(std::shared_ptr<GenericChunk>& chunk)
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
    pkt[0].length = sizeof(receive_timing);
    rc = chunk->apply_packets_layout();
    if (rc != ReturnStatus::success) {
        if (rc != ReturnStatus::signal_received) {
            std::cerr << "Error preparing a chunk to send..." << std::endl;
        }
        return rc;
    }
    return ReturnStatus::success;
}

void FrameIONode::receive_send()
{
    ReturnStatus rc = ReturnStatus::success;
    std::shared_ptr<GenericChunk> commit_chunk;
    ReceiveChunk receive_chunk(m_receive_stream->get_id(), is_receive_hds());
    m_receive_stream->set_completion_moderation(1, MAX_RX_CHUNK_SIZE, 0);
    uint64_t receive_ts = 0;

    while (rc != ReturnStatus::failure && rc != ReturnStatus::signal_received &&
                  SignalHandler::get_received_signal() < 0) {

        rc = prepare_reply_chunk(commit_chunk);
        if (rc != ReturnStatus::success) {
            break;
        }

        bool frame_start = true;
        bool frame_received = false;
        bool valid = true;
        m_rx_next_pkt_num = 0;
        m_rx_iteration = 0;
        m_rx_drop_cnt = 0;

        uint64_t first_packet_ts = 0;
        uint64_t last_packet_ts = 0;

        while (!frame_received) {
            rc = m_receive_stream->get_next_chunk(receive_chunk);
            if (rc != ReturnStatus::success) {
                if (rc != ReturnStatus::signal_received) {
                    std::cerr << "Failed to get data chunk from Rx stream" << std::endl;
                }
                return;
            }
            receive_ts = get_time_now_ns();

            if (receive_chunk.get_length() == 0) {
                continue;
            }

            parse_packet_headers(receive_chunk, frame_received, valid);

            if (frame_start) {
                first_packet_ts =
                        rmx_input_get_completion_timestamp_first(receive_chunk.get_completion());
                frame_start = 0;
            }
            if (frame_received) {
                if (!valid) {
                    std:: cerr << "drops detected :" << m_rx_drop_cnt << std::endl;
                }

                last_packet_ts =
                        rmx_input_get_completion_timestamp_last(receive_chunk.get_completion());
            }
        }

        do {
            compose_reply(commit_chunk, first_packet_ts, last_packet_ts, receive_ts,
                          m_rx_iteration, valid);
            rc = m_send_stream->commit_chunk(commit_chunk, 0);
            switch (rc) {
                case ReturnStatus::hw_send_queue_full:
                    std::this_thread::sleep_for(
                            std::chrono::microseconds(m_hw_queue_full_sleep_us));
                    break;
                case ReturnStatus::failure:
                    std::cerr << "Failed to send chunk of Tx stream" << std::endl;
                    break;
                case ReturnStatus::signal_received:
                    std::cerr << "Received signal when send chunk of Tx stream" << std::endl;
                    break;
                default:
                    break;
            }
        } while (rc == ReturnStatus::hw_send_queue_full);
    }
}
