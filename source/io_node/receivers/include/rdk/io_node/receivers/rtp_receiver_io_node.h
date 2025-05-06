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

#ifndef RDK_IO_NODE_RECEIVERS_RTP_RECEIVER_IO_NODE_H_
#define RDK_IO_NODE_RECEIVERS_RTP_RECEIVER_IO_NODE_H_

#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>
#include <chrono>
#include <string>

#include <rivermax_api.h>

#include "rdk/io_node/receivers/receiver_io_node_base.h"
#include "rdk/io_node/receivers/rtp_event_notifier_interface.h"
#include "rdk/core/flow/receive_flow.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
{
namespace io_node
{
/**
 * @brief: Application receive stream specialized to parse RTP streams.
 *
 * This class implements and extends @ref ReceiveStream operations.
 */
class AppRTPReceiveStream : public ReceiveStream
{
private:
    bool m_is_extended_sequence_number;
    uint32_t m_sequence_number_mask;
    bool m_is_header_data_split;
    bool m_is_header_processing_enabled = true;
    RXStatistics m_statistic;
    RXStatistics m_statistic_totals;
    bool m_initialized = false;
    uint32_t m_last_sequence_number = 0;
    bool m_last_mbit = true;
    std::unique_ptr<IRTPEventHandler> m_frame_notifier;
public:
    /**
     * @brief: Constructs RTP receive stream wrapper.
     *
     * @param [in] settings: Stream settings.
     * @param [in] is_extended_sequence_number: Parse extended sequence number.
     * @param [in] header_data_split: Header-data split is enabled.
     * @param [in] process_headers: Enable RTP header processing.
     */
    AppRTPReceiveStream(const ReceiveStreamSettings& settings,
        bool is_extended_sequence_number, bool header_data_split, bool process_headers = true);
    virtual ~AppRTPReceiveStream() = default;

    void print_statistics(std::ostream& out,
            const std::chrono::high_resolution_clock::duration& interval_duration) const override;
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
    RXStatistics get_statistics() const;
    /**
     * @brief: Sets the handler to events of detection of a new media frame.
     *
     * @param [in] event_handler: Pointer to event handler.
     */
    void set_frame_start_handler(std::unique_ptr<IRTPEventHandler> event_handler);
private:
    /**
     * @brief: Handles received packet.
     *
     * Parse header to extract a sequence number, count the number of dropped
     * packets by the sequence number.
     *
     * @param [in] header: Pointer to start of header.
     * @param [in] length: Packet header length.
     * @param [in] info: Packet meta-information.
     */
    void process_packet_header(const byte_t* header, size_t length, const ReceivePacketInfo& info);
protected:
    /**
     * @brief: Extracts sequence number from RTP packet and (if needed) payload header.
     *
     * If @ref m_is_extended_sequence_number is set then parse the 16 high
     * order bits of the extended 32-bit sequence number from the start of RTP
     * payload.
     *
     * @param [in] header: Pointer to start of RTP header.
     * @param [in] length: Header length.
     * @param [out] sequence_number: Sequence number.
     *
     * @return: true if packet header is valid.
     */
    bool get_sequence_number(const byte_t* header, size_t length, uint32_t& sequence_number) const;
    /**
     * @brief: Returns sequence number mask.
     *
     * @return: Sequence number mask.
     */
    uint32_t get_sequence_number_mask() const { return m_sequence_number_mask; }
};

/**
 * @brief: RTPReceiverIONode class.
 *
 * This class implements the required operations in order to be a RTP receiver.
 * The sender class will be the context that will be run under a std::thread by
 * overriding the operator (). Each receiver will be able to run multiple
 * streams.
 */
class RTPReceiverIONode : public ReceiverIONodeBase
{
private:
    std::vector<ReceiveFlow> m_flows;
public:
    /**
     * @brief: RTPReceiverNode constructor.
     *
     * @param [in] app_settings: Application settings.
     * @param [in] is_extended_sequence_number: Parse extended sequence number.
     * @param [in] devices: List of NICs to receive data.
     * @param [in] index: Receiver index.
     * @param [in] cpu_core_affinity: CPU core affinity the sender will run on.
     * @param [in] memory_utils: Memory utilities.
     */
    RTPReceiverIONode(const AppSettings& app_settings,
        bool is_extended_sequence_number,
        const std::vector<std::string>& devices,
        size_t index, int cpu_core_affinity,
        IONodeMemoryUtils& memory_utils);
    virtual ~RTPReceiverIONode() = default;

    /**
     * @brief: Initializes receive streams.
     *
     * @param [in] start_id: Starting identifier for streams list.
     * @param [in] flows: Vector of flows to be received by streams.
     */
    void initialize_streams(size_t start_id, const std::vector<ReceiveFlow>& flows);

    /**
     * @brief: Assign receive streams constructed externally.
     *
     * @param [in] start_id: Starting identifier for streams list.
     * @param [in] flows: Vector of flows to be received by streams.
     * @param [in] streams: Vector of streams to be processed by this IO node.
     */
    void assign_streams(size_t start_id, const std::vector<ReceiveFlow>& flows,
                        std::vector<std::unique_ptr<IReceiveStream>>& streams);
protected:
    std::vector<std::string> m_devices;
    bool m_is_extended_sequence_number;

    ReturnStatus attach_flows() override { return process_flows(true); }
    ReturnStatus detach_flows() override { return process_flows(false); }
    ReturnStatus consume_chunk(std::unique_ptr<IReceiveDataConsumer>& data_consumer,
        const ReceiveChunk& chunk, IReceiveStream& stream) override;

private:
    /**
     * @brief: Helper method for attaching or detaching flows to/from streams.
     *
     * Distributes flows evenly among streams, handling any remainder flows.
     * The distribution ensures each stream gets approximately the same number
     * of flows, with the first remainder flows streams getting one extra flow.
     *
     * @param [in] is_attach: If true, attaches flows to streams; if false, detaches flows from streams.
     *
     * @return: Return status of the operation.
     */
    ReturnStatus process_flows(bool is_attach);
    /**
     * @brief: Casts a unique pointer to IReceiveStream to a raw pointer to AppRTPReceiveStream.
     *
     * @param [in] stream: The unique pointer to IReceiveStream.
     *
     * @return: A raw pointer to AppRTPReceiveStream.
     */
    inline AppRTPReceiveStream* to_derived_stream(const std::unique_ptr<IReceiveStream>& stream) const {
        return static_cast<AppRTPReceiveStream*>(stream.get());
    }
};

} // io_node
} // dev_kit
} // rivermax

#endif /* RDK_IO_NODE_RECEIVERS_RTP_RECEIVER_IO_NODE_H_ */
