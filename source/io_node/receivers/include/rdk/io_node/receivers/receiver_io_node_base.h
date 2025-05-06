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

#ifndef RDK_IO_NODE_RECEIVERS_RECEIVER_IO_NODE_BASE_H_
#define RDK_IO_NODE_RECEIVERS_RECEIVER_IO_NODE_BASE_H_

#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>
#include <chrono>

#include <rivermax_api.h>

#include "rdk/io_node/common/io_node_memory_utils.h"
#include "rdk/core/data_handler/receive_data_consumer_interface.h"
#include "rdk/core/memory_layout/header_payload_memory_layout.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
{
namespace io_node
{
/**
 * @brief: Receiving statistics struct.
 *
 * This struct will hold run time statistics of a stream.
 */
struct RXStatistics {
    size_t rx_count = 0;
    size_t rx_dropped = 0;
    size_t rx_corrupt_header = 0;
    size_t received_bytes = 0;
    size_t consumed_packets = 0;
    size_t unconsumed_packets = 0;

    virtual ~RXStatistics() = default;
    /**
     * @brief: Resets values to zero.
     */
    virtual void reset() {
        rx_count = 0;
        rx_dropped = 0;
        rx_corrupt_header = 0;
        received_bytes = 0;
        consumed_packets = 0;
        unconsumed_packets = 0;
    }
    /**
     * @brief: Returns total number of packets.
     *
     * Calculate the number of packets in statistics interval (including dropped).
     *
     * @return: Total number of packets.
     */
    virtual size_t get_total_packets() const { return rx_count + rx_dropped; }
    /**
     * @brief: Returns total received data in Mbits.
     *
     * @return: Total received data in Mbits.
     */
    virtual double get_Mbits() const { return ((received_bytes * 8) / 1.e6); }
};
/**
 * @brief: ReceiverIONodeBase class.
 *
 * This class implements the required operations in order to be a receiver.
 * The receiver class will be the context that will be run under a std::thread by
 * overriding the operator (). Each receiver will be able to run multiple
 * streams.
 */
class ReceiverIONodeBase : public IHeaderPayloadMemoryLayoutComponent
{
protected:
    static constexpr size_t DEFAULT_MAX_CHUNK_SIZE = 1024;
    AppSettings m_app_settings;
    size_t m_index;
    bool m_print_parameters;
    int m_cpu_core_affinity;
    std::chrono::microseconds m_sleep_between_operations;
    IONodeMemoryUtils& m_memory_utils;
    std::vector<std::unique_ptr<IReceiveStream>> m_streams;
    std::vector<std::unique_ptr<ReceiveChunk>> m_stream_chunks;
    std::vector<std::unique_ptr<IReceiveDataConsumer>> m_data_consumers;
    size_t m_header_total_memory_size = 0;
    size_t m_payload_total_memory_size = 0;
    std::vector<std::pair<size_t, size_t>> m_aligned_header_payload_buffers_size;
    uint32_t m_print_interval_ms = 1000;

public:
    virtual ~ReceiverIONodeBase() = default;
    ReturnStatus initialize_memory_layout() override;
    ReturnStatus determine_memory_layout(HeaderPayloadMemoryLayoutRequest& memory_layout_request) const override;
    ReturnStatus apply_memory_layout(const HeaderPayloadMemoryLayoutResponse& memory_layout_response) override;
    ReturnStatus validate_memory_layout(const HeaderPayloadMemoryLayoutResponse& memory_layout_respose) const override;
    /**
     * @brief: Prints receiver's parameters to a output stream.
     *
     * The method prints the parameters of the receiver to be shown to the user
     * to a output stream.
     *
     * @param [out] out: Output stream parameter print to.
     *
     * @return: Output stream.
     */
    virtual std::ostream& print(std::ostream& out) const;
    /**
     * @brief: Overrides operator << for @ref ReceiverIONodeBase reference.
     */
    friend std::ostream& operator<<(std::ostream& out, const ReceiverIONodeBase& receiver)
    {
        receiver.print(out);
        return out;
    }
    /**
     * @brief: Overrides operator << for @ref ReceiverIONodeBase pointer.
     */
    friend std::ostream& operator<<(std::ostream& out, ReceiverIONodeBase* receiver)
    {
        receiver->print(out);
        return out;
    }
    /**
     * @brief: Prints receiver's parameters.
     *
     * @note: The information will be printed if the receiver was initialized with
     *        @ref app_settings->print_parameters parameter of set to true.
     */
    virtual void print_parameters() const;
    /**
     * @brief: Sets statistics report interval.
     *
     * @param [in] print_interval_ms: Print interval in milliseconds. Set 0 to disable.
     */
    void set_statistics_report_interval(uint32_t print_interval_ms);
    /**
     * @brief: Returns receiver index.
     *
     * @return: Receiver index.
     */
    virtual size_t get_index() const { return m_index; }
    /**
     * @brief: Sets the receive data consumer for the specified stream index.
     *
     * @param [in] stream_index: Stream index.
     * @param [in] data_consumer: Receive data consumer pointer.
     *
     * @return: Status of the operation.
     */
    ReturnStatus set_receive_data_consumer(size_t stream_index, std::unique_ptr<IReceiveDataConsumer> data_consumer);
    /**
     * @brief: Returns streams total statistics.
     *
     * @tparam StatisticsType: Type of the statistics.
     * @tparam StreamType: Type of the stream.
     *
     * @return: Vector of stream statistics.
     */
    template<typename StatisticsType, typename StreamType>
    std::vector<StatisticsType> get_streams_total_statistics() const;
    /**
     * @brief: Receiver's worker.
     *
     * This method is the worker method of the std::thread will run with this
     * object as it's context.
     */
    virtual void operator()();
protected:
    /**
     * @brief: ReceiverIONodeBase constructor.
     *
     * @param [in] app_settings: Application settings.
     * @param [in] index: Receiver index.
     * @param [in] cpu_core_affinity: CPU core affinity the receiver will run on.
     * @param [in] memory_utils: Memory utilities.
     */
    ReceiverIONodeBase(const AppSettings& app_settings,
        size_t index, int cpu_core_affinity, IONodeMemoryUtils& memory_utils);
    /**
     * @brief: Creates receiver's streams.
     *
     * This method is responsible to go over receiver's stream objects and
     * create the appropriate Rivermax streams.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus create_streams();
    /**
     * @brief: Destroys receiver's streams.
     *
     * This method is responsible to go over receiver's stream objects and
     * destroy the appropriate Rivermax stream.
     *
     * @return: ReturnStatus::success if the memory configuration is set successfully,
     *          otherwise ReturnStatus::failure.
     */
    virtual ReturnStatus destroy_streams();
    /**
     * @brief: Sets CPU related resources.
     *
     * This method is responsible to set receiver's priority and CPU core affinity.
     */
    virtual void set_cpu_resources();
    /**
     * @brief: Prints stream statistics.
     *
     * @param [out] out: Output stream to print statistics to.
     * @param [in] interval_duration: Statistics interval duration.
     */
    virtual void print_statistics(std::ostream& out,
        const std::chrono::high_resolution_clock::duration& interval_duration) const;
    /**
     * @brief: Returns a chunk that belongs to the specified stream for receiving data.
     *
     * @param [in] stream_index: The internal stream index.
     *
     * @return: A reference to the chunk.
     */
    virtual ReceiveChunk& get_stream_chunk(size_t stream_index) const;
    /**
     * @brief: Creates a new chunk for receiving data.
     *
     * @param [in] stream_index: Internal stream index.
     *
     * @return: A unique pointer to the created chunk.
     */
    virtual std::unique_ptr<ReceiveChunk> create_chunk(size_t stream_index) const;
    /**
     * @brief: Applies a memory layout for the stream.
     *
     * This function applies a memory layout for the subcomponent (stream) based on the provided
     * memory layout and aligned stream buffer sizes.
     *
     * @param [in] stream: The stream to build the memory layout for.
     * @param [in] provided_memory_layout: The memory layout provided to the IO Node.
     * @param [in] stream_header_payload_buffers_size: The aligned stream header payload buffer sizes.
     * @param [out] header_offset: The offset for the header memory.
     * @param [out] payload_offset: The offset for the payload memory.
     *
     * @return: Status of the operation.
     */
    ReturnStatus apply_memory_layout_for_subcomponent(
        std::unique_ptr<IReceiveStream>& stream,
        const HeaderPayloadMemoryLayout& provided_memory_layout,
        const std::pair<size_t, size_t>& stream_header_payload_buffers_size,
        size_t& header_offset, size_t& payload_offset) const;
    /**
     * @brief: Returns whether the memory layout is initialized.
     *
     * @return: True if the memory layout is initialized, otherwise false.
     */
    bool is_memory_layout_initialized() const { return m_payload_total_memory_size > 0; }
    /**
     * @brief: Attaches flows to receiver's streams.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus attach_flows() = 0;
     /**
      * @brief: Detaches flows from receiver's streams.
      *
      * @return: Status of the operation.
      */
    virtual ReturnStatus detach_flows() = 0;
    /**
     * @brief: Process a received chunk.
     *
     * @param [in] data_consumer: Data consumer.
     * @param [in] chunk: Received chunk.
     * @param [in] stream: Receive stream.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus consume_chunk(std::unique_ptr<IReceiveDataConsumer>& data_consumer,
        const ReceiveChunk& chunk, IReceiveStream& stream);
    /**
     * @brief: Wait for a first input packet.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus wait_first_packet();
    /**
     * @brief: Start receiving streams synchronously.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus synchronous_start() { return wait_first_packet(); };
};

template<typename StatisticsType, typename StreamType>
std::vector<StatisticsType> ReceiverIONodeBase::get_streams_total_statistics() const
{
    std::vector<StatisticsType> streams_stats;

    for (const auto& stream : m_streams) {
        streams_stats.push_back(static_cast<StreamType*>(stream.get())->get_statistics());
    }

    return streams_stats;
}

} // io_node
} // dev_kit
} // rivermax

#endif /* RDK_IO_NODE_RECEIVERS_RECEIVER_IO_NODE_BASE_H_ */
