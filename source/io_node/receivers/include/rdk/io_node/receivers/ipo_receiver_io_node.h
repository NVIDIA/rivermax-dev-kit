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

#ifndef RDK_IO_NODE_RECEIVERS_IPO_RECEIVER_IO_NODE_H_
#define RDK_IO_NODE_RECEIVERS_IPO_RECEIVER_IO_NODE_H_

#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>
#include <chrono>

#include <rivermax_api.h>

#include "rdk/io_node/receivers/rtp_receiver_io_node.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
{
namespace io_node
{
/**
 * @brief: Receives path statistics.
 */
struct IPOPathStatistics
{
    uint32_t rx_count = 0;
    uint32_t rx_dropped = 0;

    /**
     * @brief: Resets values to zero.
     */
    void reset()
    {
        rx_count = 0;
        rx_dropped = 0;
    }
};
/**
 * @brief: Receiving statistics struct.
 *
 * This struct will hold run time statistics of a stream.
 */
struct IPORXStatistics : public RXStatistics
{
    size_t rx_exceed_md = 0;
    std::vector<IPOPathStatistics> path_stats;

    void reset() override
    {
        RXStatistics::reset();
        rx_exceed_md = 0;

        for (auto& path_stat : path_stats) {
            path_stat.reset();
        }
    }
};
/**
 * @brief: Application IPO receive stream specialized to parse RTP streams.
 *
 * This class implements and extends @ref IPOReceiveStream operations.
 */
class AppIPOReceiveStream : public IPOReceiveStream
{
private:
    bool m_is_extended_sequence_number;
    uint32_t m_sequence_number_mask;

    IPORXStatistics m_statistic;
    IPORXStatistics m_statistic_totals;
    std::vector<std::vector<uint8_t>> m_path_packets;
    bool m_initialized = false;
    uint32_t m_last_sequence_number = 0;
public:
    /**
     * @brief: Constructs Inline Packet Ordering stream wrapper.
     *
     * @param [in] settings: Stream settings.
     * @param [in] is_extended_sequence_number: Parse extended sequence number.
     * @param [in] paths: List of redundant data receive paths.
     */
    AppIPOReceiveStream(const ipo_stream_settings_t& settings,
            bool is_extended_sequence_number, const std::vector<IPOReceivePath>& paths);
    virtual ~AppIPOReceiveStream() = default;

    void print_statistics(std::ostream& out, const std::chrono::high_resolution_clock::duration& interval_duration) const override;
    void reset_statistics() override;
    void reset_statistics_totals() override;
    ReturnStatus get_next_chunk(ReceiveChunk& chunk) final;
    /**
    * @brief: Updates the consumed packets statistics.
    *
    * @param [in] consumed_packets: Number of consumed packets.
    * @param [in] unconsumed_packets: Number of unconsumed packets.
    */
    void update_consumed_packets_stats(size_t consumed_packets, size_t unconsumed_packets);
    /**
     * @brief: Returns stream statistics.
     *
     * @return: Stream statistics.
     */
    IPORXStatistics get_statistics() const;
private:
    void handle_corrupted_packet(size_t index, const ReceivePacketInfo& packet_info) final;
    void handle_late_packet(size_t index, uint32_t sequence_number, const ReceivePacketInfo& packet_info) override;
    void handle_packet(size_t index, uint32_t sequence_number, const ReceivePacketInfo& packet_info) final;
    void handle_redundant_packet(size_t index, uint32_t sequence_number, const ReceivePacketInfo& packet_info) final;
    void complete_packet(uint32_t sequence_number) final;
    void handle_sender_restart() override;

protected:
    bool get_sequence_number(const byte_t* header, size_t length, uint32_t& sequence_number) const final;
    uint32_t get_sequence_number_mask() const final { return m_sequence_number_mask; }
};

/**
 * @brief: IPOReceiverIONode class.
 *
 * This class implements the required operations in order to be a IPO receiver.
 * The sender class will be the context that will be run under a std::thread by
 * overriding the operator (). Each receiver will be able to run multiple
 * streams.
 */
class IPOReceiverIONode : public RTPReceiverIONode
{
private:
    ipo_stream_settings_t m_stream_settings;
public:
    /**
     * @brief: IPOReceiverNode constructor.
     *
     * @param [in] app_settings: Application settings.
     * @param [in] max_path_differential_us: Maximum Path Differential value.
     * @param [in] is_extended_sequence_number: Parse extended sequence number.
     * @param [in] devices: List of NICs to receive data.
     * @param [in] index: Receiver index.
     * @param [in] cpu_core_affinity: CPU core affinity the sender will run on.
     * @param [in] memory_utils: Memory utilities.
     */
    IPOReceiverIONode(const AppSettings& app_settings,
        uint64_t max_path_differential_us,
        bool is_extended_sequence_number,
        const std::vector<std::string>& devices,
        size_t index, int cpu_core_affinity,
        IONodeMemoryUtils& memory_utils);
    virtual ~IPOReceiverIONode() = default;
    /**
     * @brief: Initializes receive streams.
     *
     * @param [in] start_id: Starting identifier for streams list.
     * @param [in] flows: Vector of vectors of flows to be received by streams.
     *                    Each item in outer level vector must contain vector
     *                    of the same number of items as in devices list passed
     *                    into constructor. Each item in the inner vector will
     *                    be mapped to a corresponding device.
     */
    void initialize_streams(size_t start_id, const std::vector<std::vector<ReceiveFlow>>& flows);

private:
    ReturnStatus attach_flows() override;
    virtual ReturnStatus synchronous_start() override;
    ReturnStatus detach_flows() override;
    ReturnStatus consume_chunk(std::unique_ptr<IReceiveDataConsumer>& data_consumer,
        const ReceiveChunk& chunk, IReceiveStream& stream) override;
    std::unique_ptr<ReceiveChunk> create_chunk(size_t stream_index) const override;
    /**
     * @brief: Sync all streams.
     *
     * Flushes input buffers of all streams.
     *
     * @return: Status of the operation.
     */
    ReturnStatus sync_streams();
    /**
     * @brief: Casts a unique pointer to IReceiveStream to a raw pointer to AppIPOReceiveStream.
     *
     * @param [in] stream: The unique pointer to IReceiveStream.
     *
     * @return: A raw pointer to AppIPOReceiveStream.
     */
    inline AppIPOReceiveStream* to_derived_stream(const std::unique_ptr<IReceiveStream>& stream) const {
        return static_cast<AppIPOReceiveStream*>(stream.get());
    }
};

} // io_node
} // dev_kit
} // rivermax

#endif /* RDK_IO_NODE_RECEIVERS_IPO_RECEIVER_IO_NODE_H_ */
