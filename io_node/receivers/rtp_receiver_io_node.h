/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef RMAX_APPS_LIB_IO_NODE_RECEIVERS_RTP_RECEIVER_IO_NODE_H_

#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>
#include <chrono>

#include <rivermax_api.h>

#include "api/rmax_apps_lib_api.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

namespace ral
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
    size_t rx_corrupt_rtp_header = 0;
    size_t received_bytes = 0;

    /**
     * @brief: Resets values to zero.
     */
    void reset()
    {
        rx_count = 0;
        rx_dropped = 0;
        rx_corrupt_rtp_header = 0;
        received_bytes = 0;
    }
    /**
     * @brief: Returns total number of packets.
     *
     * Calculate the number of packets in statistics interval (including
     * dropped as calculated by sequence number differences).
     *
     * @return: Returns total number of packets.
     */
    size_t get_total_packets() const {
        return rx_count + rx_dropped;
    }
    /**
     * @brief: Returns total received data in Mbits.
     *
     * @return: Returns total received data in Mbits.
     */
    double get_Mbits() const
    {
        return ((received_bytes * 8) / 1.e6);
    }
};

/**
 * @brief: Application receive stream specialized to parse RTP streams.
 *
 * This class implements and extends @ref ral::lib::core::ReceiveStream operations.
 */
class AppRTPReceiveStream : public ReceiveStream
{
private:
    const size_t m_stream_index;
    const bool m_is_extended_sequence_number;
    const uint32_t m_sequence_number_mask;
    const bool m_is_header_data_split;

    RXStatistics m_statistic;
    bool m_initialized = false;
    uint32_t m_last_sequence_number = 0;

public:
    /**
     * @brief: Constructs RTP receive stream wrapper.
     *
     * @param [in] id: Stream identifier.
     * @param [in] settings: Stream settings.
     * @param [in] extended_sequence_number: Parse extended sequence number.
     * @param [in] header_data_split: Header-data split is enabled.
     */
    AppRTPReceiveStream(size_t id, const ReceiveStreamSettings& settings,
            bool extended_sequence_number, bool header_data_split);
    virtual ~AppRTPReceiveStream() = default;

    /**
     * @brief: Prints stream statistics.
     *
     * @param [out] out: Output stream to print statistics to.
     * @param [in] interval_duration: Statistics interval duration.
     */
    void print_statistics(std::ostream& out,
            const std::chrono::high_resolution_clock::duration& interval_duration) const;
    /**
     * @brief: Resets statistics.
     */
    void reset_statistics();
    /**
     * @brief: Receives next chunk from input stream.
     *
     * @param [out] chunk: Pointer to the returned chunk structure.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::signal_received - If operation was interrupted by an OS signal.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus get_next_chunk(ReceiveChunk& chunk) final;

private:
    /**
     * @brief: Handles received packet.
     *
     * Parse header to extract a sequence number, count the number of dropped
     * packets by the sequence number.
     *
     * @param [in] index: Redundant stream index (0-based).
     * @param [in] sequence_number: Sequence number.
     * @param [in] packet_info: Detailed packet information.
     */
    void process_packet(const byte_t* header, size_t length);

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
     * @brief: Gets sequence number mask.
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
class RTPReceiverIONode
{
private:
    static constexpr size_t DEFAULT_MAX_CHUNK_SIZE = 1024;
    const AppSettings m_app_settings;
    const bool m_is_extended_sequence_number;
    const std::string m_device;
    const size_t m_index;
    const bool m_print_parameters;
    const int m_cpu_core_affinity;
    const std::chrono::microseconds m_sleep_between_operations;

    std::vector<std::unique_ptr<AppRTPReceiveStream>> m_streams;
    std::vector<FourTupleFlow> m_flows;

public:
    /**
     * @brief: RTPReceiverNode constructor.
     *
     * @param [in] app_settings: Application settings.
     * @param [in] extended_sequence_number: Parse extended sequence number.
     * @param [in] devices: List of NICs to receive data.
     * @param [in] index: Receiver index.
     * @param [in] cpu_core_affinity: CPU core affinity the sender will run on.
     */
    RTPReceiverIONode(const AppSettings& app_settings,
        bool extended_sequence_number,
        const std::string& device,
        size_t index, int cpu_core_affinity);
    virtual ~RTPReceiverIONode() = default;
    /**
     * @brief: Returns receiver's streams container.
     *
     * @return: Receiver's streams container.
     */
    std::vector<std::unique_ptr<AppRTPReceiveStream>>& get_streams() { return m_streams; }
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
    std::ostream& print(std::ostream& out) const;
    /**
     * @brief: Overrides operator << for @ref ral::io_node::RTPReceiverIONode reference.
     */
    friend std::ostream& operator<<(std::ostream& out, const RTPReceiverIONode& receiver)
    {
        receiver.print(out);
        return out;
    }
    /**
     * @brief: Overrides operator << for @ref ral::io_node::RTPReceiverIONode pointer.
     */
    friend std::ostream& operator<<(std::ostream& out, RTPReceiverIONode* receiver)
    {
        receiver->print(out);
        return out;
    }
    /**
     * @brief: Initializes receive streams.
     *
     * @param [in] start_id: Starting identifier for streams list.
     * @param [in] flows: Vector of flows to be received by streams.
     */
    void initialize_streams(size_t start_id, const std::vector<FourTupleFlow>& flows);
    /**
     * @brief: Prints receiver's parameters.
     *
     * @note: The information will be printed if the receiver was initialized with
     *         @ref app_settings->print_parameters parameter of set to true.
     */
    void print_parameters();
    /**
     * @brief: Returns receiver index.
     *
     * @return: Receiver index.
     */
    size_t get_index() const { return m_index; }
    /**
     * @brief: Receiver's worker.
     *
     * This method is the worker method of the std::thread will run with this
     * object as it's context. The user of @ref ral::io_node::RTPReceiverIONode
     * class can initialize the object in advance and run std::thread when
     * needed.
     */
    void operator()();
private:
    /**
     * @brief: Creates receiver's streams.
     *
     * This method is responsible to go over receiver's stream objects and
     * create the appropriate Rivermax streams.
     *
     * @return: Status of the operation.
     */
    ReturnStatus create_streams();
    /**
     * @brief: Attaches flows to receiver's streams.
     *
     * @return: Status of the operation.
     */
    ReturnStatus attach_flows();
    /**
     * @brief: Wait for a first input packet.
     *
     * @return: Status of the operation.
     */
     ReturnStatus wait_first_packet();
    /**
     * @brief: Detaches flows from receiver's streams.
     *
     * @return: Status of the operation.
     */
    ReturnStatus detach_flows();
    /**
     * @brief: Destroys receiver's streams.
     *
     * This method is responsible to go over receiver's stream objects and
     * destroy the appropriate Rivermax stream.
     *
     * @return: Status of the operation.
     */
    ReturnStatus destroy_streams();
    /**
     * @brief: Sets CPU related resources.
     *
     * This method is responsible to set sender's priority and CPU core affinity.
     */
    void set_cpu_resources();
};

} // io_node
} // ral

#define RMAX_APPS_LIB_IO_NODE_RECEIVERS_RTP_RECEIVER_IO_NODE_H_
#endif /* RMAX_APPS_LIB_IO_NODE_RECEIVERS_RTP_RECEIVER_IO_NODE_H_ */
