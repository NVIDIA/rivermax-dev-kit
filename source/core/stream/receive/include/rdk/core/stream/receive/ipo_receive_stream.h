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

#ifndef RDK_CORE_STREAM_RECEIVE_IPO_RECEIVE_STREAM_H_
#define RDK_CORE_STREAM_RECEIVE_IPO_RECEIVE_STREAM_H_

#include <chrono>
#include <ostream>
#include <string>
#include <vector>
#include <unordered_set>

#include <rivermax_api.h>

#include "rdk/core/stream/receive/receive_stream.h"
#include "rdk/core/stream/receive/aggregate_receive_stream_interface.h"
#include "rdk/core/chunk/ipo_receive_chunk.h"

namespace rivermax
{
namespace dev_kit
{
namespace core
{

/**
 * @brief: Redundant stream settings.
 */
typedef struct ipo_stream_settings
{
    uint16_t packet_payload_size = 0;
    uint16_t packet_app_header_size = 0;
    size_t num_of_packets_in_chunk = 0;
    size_t max_chunk_size = 0;
    std::unordered_set<rmx_input_option> stream_options;
    uint64_t max_path_differential_us = 0;

    ipo_stream_settings()
    {
    }

    ipo_stream_settings(
            uint16_t packet_payload_size_,
            uint16_t packet_app_header_size_,
            size_t num_of_packets_in_chunk_,
            size_t max_chunk_size_,
            const std::unordered_set<rmx_input_option>& stream_options_,
            uint64_t max_path_differential_us_
    ) :
        packet_payload_size(packet_payload_size_),
        packet_app_header_size(packet_app_header_size_),
        num_of_packets_in_chunk(num_of_packets_in_chunk_),
        max_chunk_size(max_chunk_size_),
        stream_options(stream_options_),
        max_path_differential_us(max_path_differential_us_)
    {
    }

} ipo_stream_settings_t;

/**
 * @brief: This structure describes a receive path (NIC + network flow).
 */
struct IPOReceivePath
{
    std::string dev_ip;
    ReceiveFlow flow;

    IPOReceivePath(const std::string& dev_ip_, const ReceiveFlow& flow_) :
        dev_ip(dev_ip_),
        flow(flow_)
    {
    }
};

/**
 * @brief: Inline Packet Ordering receiver class.
 *
 * This class implements redundant multi-path receive stream by
 * incapsulating multiple streams using Inline Packet Ordering feature over the
 * same memory buffer.
 */
class IPOReceiveStream : public IAggregateReceiveStream
{
private:
    using clock = std::chrono::steady_clock;
    static_assert(clock::is_steady, "clock must be steady");

    /**
     * @brief: Extended packet information to be used in stream reconstruction.
     */
    struct ext_packet_info {
        uint32_t sequence_number = 0;
        bool is_valid = false;
        clock::time_point timestamp;
        uint64_t hw_timestamp = 0;
    };

    /**
     * @brief: State of stream reconstruction code.
     */
    enum class State {
        /**
         * Stream is not started yet.
         * Switch to @ref State::WaitFirstPacket on @ref start call.
         */
        NotStarted,
        /**
         * Waiting for the first input packet to initialize receiver.
         * Switch to @ref State::Running on a first packet.
         */
        WaitFirstPacket,
        /**
         * Buffer is non-empty. We're waiting for buffered packets processing
         * time. Switching to @ref State::Waiting state once the buffer is empty.
         */
        Running,
        /**
         * Waiting for the next input packet. Transition to @ref
         * State::Running state on packet arrival.
         */
        Waiting
    };

protected:
    ipo_stream_settings_t m_settings;
    std::vector<IPOReceivePath> m_paths;
    bool m_use_ext_seqn;
    bool m_header_data_split;
    bool m_pkt_info_enabled;
    uint32_t m_num_of_packets_in_chunk;

    std::chrono::microseconds m_max_path_differential;
    std::chrono::microseconds m_sender_restart_threshold{ std::chrono::milliseconds(100) };

    uint32_t m_last_processed_sequence_number = 0;
    uint32_t m_sequence_number_wrap_around = 0;
    uint32_t m_sequence_number_msb_mask = 0;
    uint32_t m_sequence_number_init_offset = 0;
    State m_state = State::NotStarted;
    std::vector<ReceiveStream> m_streams;
    std::vector<ReceiveChunk> m_chunks;
    byte_t* m_header_buffer = nullptr;
    byte_t* m_payload_buffer = nullptr;
    std::vector<ext_packet_info> m_ext_packet_info_arr;
    std::vector<ReceivePacketInfo> m_packet_info_arr;
    size_t m_header_stride_size = 0;
    size_t m_payload_stride_size = 0;
    size_t m_header_buffer_size = 0;
    size_t m_payload_buffer_size = 0;
    size_t m_index = 0;
    clock::time_point m_next_packet_time;
    clock::time_point m_start_time;
    clock::time_point m_now;
public:
    /**
     * @brief: Constructs Inline Packet Ordering stream.
     *
     * @param [in] settings: Stream settings.
     * @param [in] paths: List of redundant data receive paths.
     * @param [in] placement_order: Packet placement order (Rivermax constant).
     */
    IPOReceiveStream(const ipo_stream_settings_t& settings,
            const std::vector<IPOReceivePath>& paths, bool use_ext_seqn);

    virtual ~IPOReceiveStream() = default;
    std::ostream& print(std::ostream& out) const override;
    ReturnStatus create_stream() override;
    ReturnStatus detach_flows() override;
    ReturnStatus destroy_stream() override;
    ReturnStatus get_next_chunk(ReceiveChunk& chunk)  override;
    ReturnStatus initialize_memory_layout() override;
    ReturnStatus determine_memory_layout(HeaderPayloadMemoryLayoutRequest& memory_layout_request) const override;
    ReturnStatus apply_memory_layout(const HeaderPayloadMemoryLayoutResponse& memory_layout_response) override;
    ReturnStatus validate_memory_layout(const HeaderPayloadMemoryLayoutResponse& memory_layout_respose) const override;
    size_t get_header_stride_size() const override { return !m_streams.empty() ? m_streams[0].get_header_stride_size() : 0; }
    size_t get_payload_stride_size() const override { return !m_streams.empty() ? m_streams[0].get_payload_stride_size() : 0; }
    bool is_header_data_split_on() const override { return m_header_data_split; }
    rmx_stream_id get_id() const override { return !m_streams.empty() ? m_streams[0].get_id() : 0; }
    /**
     * @brief: Attaches flows to all redundant streams.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus attach_flows();
    /**
     * @brief: Synchronize all stream paths.
     *
     * Empties input buffers in all paths.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus sync_paths();
    /**
     * @brief: Mark current time as stream start time.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    virtual void start();
protected:
    /**
     * @brief: Handles packet with corrupt RTP header.
     *
     * @param [in] index: Redundant stream index (0-based).
     * @param [in] packet_info: Detailed packet information.
     */
    virtual void handle_corrupted_packet(size_t index, const ReceivePacketInfo& packet_info);
    /**
     * @brief: Handles packet that arrived too late.
     *
     * @param [in] index: Redundant stream index (0-based).
     * @param [in] sequence_number: RTP sequence number.
     * @param [in] packet_info: Detailed packet information.
     */
    virtual void handle_late_packet(size_t index, uint32_t sequence_number, const ReceivePacketInfo& packet_info);
    /**
     * @brief: Handles received packet.
     *
     * This function is called only for the first packet, for redundant packets
     * copies received from another streams @ref handle_redundant_packet will
     * be called.
     *
     * @param [in] index: Redundant stream index (0-based).
     * @param [in] sequence_number: RTP sequence number.
     * @param [in] packet_info: Detailed packet information.
     */
    virtual void handle_packet(size_t index, uint32_t sequence_number, const ReceivePacketInfo& packet_info);
    /**
     * @brief: Handles received redundant packet.
     *
     * This function is called only for redundant packet(s), for the first
     * received packet @ref handle_packet will be called.
     *
     * @param [in] index: Redundant stream index (0-based).
     * @param [in] sequence_number: RTP sequence number.
     * @param [in] packet_info: Detailed packet information.
     */
    virtual void handle_redundant_packet(size_t index, uint32_t sequence_number, const ReceivePacketInfo& packet_info);
    /**
     * @brief: Handles packet before returning it to caller.
     *
     * This function is called when packet is transferred from cache buffer to
     * the caller.
     *
     * @param [in] sequence_number: RTP sequence number.
     */
    virtual void complete_packet(uint32_t sequence_number);
    /**
     * @brief: Handles sender restart.
     *
     * This function is called once receiver detects that the sender restarted streaming.
     */
    virtual void handle_sender_restart();
    /**
     * @brief: Returns wrap-around value for sequence number.
     *
     * @return: Wrap-around value for sequence number.
     */
    uint32_t get_sequence_number_wrap_around() const { return m_sequence_number_wrap_around; }
    /**
     * @brief: Parses sequence number from the packet header.
     *
     * @param [in] header: Pointer to the start of the packet header.
     * @param [in] length: Header length.
     * @param [out] sequence_number: Packet sequence number.
     *
     * @return: True on success.
     */
    virtual bool get_sequence_number(const byte_t* header, size_t length, uint32_t& sequence_number) const = 0;
    /**
     * @brief: Returns sequence number mask.
     *
     * @return: Sequence number mask.
     */
    virtual uint32_t get_sequence_number_mask() const = 0;
private:
    /**
     * @brief: Initializes sub-streams for all receive paths.
     */
    void initialize_substreams();
    /**
     * @brief: Handles IO completion returned from substream.
     *
     * @param [in] index: Redundant stream index (0-based).
     * @param [in] stream: Redudnant stream.
     * @param [in] chunk: Received chunk.
     */
    inline void process_completion(size_t index, const ReceiveStream& stream, ReceiveChunk& chunk);
    /**
     * @brief: Returns wrap-round value for sequence number in circular queue of
     * @ref buffer_elements elements.
     *
     * @param [in] buffer_elements: Number of elements in circular buffer.
     *
     * @return: Wrap-around value.
     */
    inline uint32_t get_sequence_number_wrap_around(uint32_t buffer_elements) const;
};

} // namespace core
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_CORE_STREAM_RECEIVE_IPO_RECEIVE_STREAM_H_ */
