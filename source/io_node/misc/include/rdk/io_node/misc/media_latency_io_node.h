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

#ifndef RDK_IO_NODE_MISC_MEDIA_LATENCY_IO_NODE_H_
#define RDK_IO_NODE_MISC_MEDIA_LATENCY_IO_NODE_H_

#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>

#include <rivermax_api.h>

#include "rdk/io_node/misc/generic_latency_io_node.h"
#include "rdk/io_node/common/rtp_video_send_stream.h"
#include "rdk/core/stream/send/generic_stream.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
{
namespace io_node
{

#pragma pack(push, 1)
struct MediaRxLatencyReply {
    int64_t rx_delay_min;
    int64_t rx_delay_avg;
    int64_t rx_delay_max;
    uint64_t packets_lost;
};
#pragma pack(pop)

/**
 * @brief: MediaTxIONode class.
 *
 * This class implements a video stream sender and for sent data packets measures the delay
 * between the time when the packet should be sent according SMPTE 2110-21 standard with
 * the given stream settings and the time when the packet is sent (Tx completion timestamp).
 * Due to Tx completion limitations the time is measured only for the last packet in the chunk.
 *
 */
class MediaTxIONode: public LatencyIONode
{
protected:
    std::shared_ptr<AppSettings> m_app_settings;
    std::shared_ptr<RtpVideoSendStream> m_send_stream;
    std::shared_ptr<ReceiveStream> m_receive_stream;
    StreamDimensions m_receive_dim;
    uint32_t m_hw_queue_full_sleep_us;
    size_t m_send_data_stride_size;
    size_t m_send_header_stride_size;
    std::unique_ptr<MediaStreamMemBlockset> m_send_mem_blockset;
    std::vector<uint16_t>m_send_block_header_sizes;
    std::vector<uint16_t>m_send_block_payload_sizes;
public:
    /**
     * @brief: MediaLatencyIONode constructor.
     *
     * @param [in] settings: Latency measurement settings.
     * @param [in] header_mem_utils: Header Memory utilities.
     * @param [in] payload_mem_utils: Payload Memory utilities.
     * @param [in] time_handler_cb: Timer handler.
     */
    MediaTxIONode(
        const LatencyNodeSettings& settings,
        std::shared_ptr<MemoryUtils> header_mem_utils,
        std::shared_ptr<MemoryUtils> payload_mem_utils,
        time_handler_ns_cb_t time_handler_cb);
    virtual ~MediaTxIONode() = default;
    std::ostream& print(std::ostream& out) const;
    /**
     * @brief: Overrides operator << for @ref MediaTxIONode reference.
     */
    friend std::ostream& operator<<(std::ostream& out, const MediaTxIONode& sender)
    {
        sender.print(out);
        return out;
    }
    /**
     * @brief: Overrides operator << for @ref GenericLatencyIONode pointer.
     */
    friend std::ostream& operator<<(std::ostream& out, MediaTxIONode* sender)
    {
        sender->print(out);
        return out;
    }
    void initialize_send_stream() override;
    void initialize_receive_stream(const TwoTupleFlow& flow) override;
    ReturnStatus query_memory_size(size_t& tx_header_size, size_t& tx_payload_size,
                                   size_t& rx_header_size, size_t& rx_payload_size) override;
    void distribute_memory_for_streams(rmx_mem_region& tx_header_mreg,
                                       rmx_mem_region& tx_payload_mreg,
                                       rmx_mem_region& rx_header_mreg,
                                       rmx_mem_region& rx_payload_mreg) override;
    void print_parameters() override;
protected:
    void prepare_send_buffer() override {};
    ReturnStatus create_send_stream() override;
    ReturnStatus destroy_send_stream() override;
    ReturnStatus create_receive_stream() override;
    ReturnStatus destroy_receive_stream() override;
    ReturnStatus attach_receive_flow() override;
    virtual ReturnStatus detach_receive_flow() override;
    void send_receive() override;
    void receive_send() override {}; /* This class cannot be a server */
    /**
     * @brief: Waits for the next frame.
     *
     * This method implements logic to wait and wake up when next frame send time is close.
     *
     * @param [in] send_time_ns: Send time of the next frame in nanoseconds.
     */
    inline void wait_for_next_frame(uint64_t send_time_ns);
    /**
     * @brief: Parses the server reply.
     *
     * @param [out] timing: Measured latency data.
     *
     * @return: True if reply is valid.
     */
    bool parse_receive_timing(ReceiveChunk& chunk, MediaRxLatencyReply& timing);
    ReturnStatus try_process_one_completion(LatencyStats& tx_delay);
private:
    std::vector<uint64_t> m_commit_ts;
    double m_start_send_time_ns;
    uint64_t m_marked_token;
    uint64_t m_handled_token;
    double m_trs;
    std::unique_ptr<MediaChunk> m_chunk_handler;
};

class MediaRxIONode: public GenericLatencyIONode
{
public:
    /**
     * @brief: FrameIONode constructor.
     *
     * @param [in] settings: Latency measurement settings.
     * @param [in] header_mem_utils: Header Memory utilities.
     * @param [in] payload_mem_utils: Payload Memory utilities.
     * @param [in] time_handler_cb: Timer handler.
     */
    MediaRxIONode(
        const LatencyNodeSettings& settings,
        std::shared_ptr<MemoryUtils> header_mem_utils,
        std::shared_ptr<MemoryUtils> payload_mem_utils,
        time_handler_ns_cb_t time_handler_cb);
    virtual ~MediaRxIONode() {};
    /**
     * @brief: Overrides operator << for @ref MediaRxIONode reference.
     */
    friend std::ostream& operator<<(std::ostream& out, const MediaRxIONode& sender)
    {
        sender.print(out);
        return out;
    }
    /**
     * @brief: Overrides operator << for @ref MediaRxIONode pointer.
     */
    friend std::ostream& operator<<(std::ostream& out, MediaRxIONode* sender)
    {
        sender->print(out);
        return out;
    }
protected:
    std::shared_ptr<AppSettings> m_app_settings;

    void prepare_send_buffer() override {}
    void send_receive() override {} /* This class cannot be a client */
    void receive_send() override;
    /**
     * @brief: Returns RTP parameters of the frame.
     *
     * @param [in] rtp_hdr: Pointer to packet's RTP header.
     * @param [out] sequence_number: RTP sequence number.
     * @param [out] timestamp: RTP timestamp.
     * @param [in] m_bit: M-bit (end of field).
     * @param [in] f_bit: F-bit (filed index).
     *
     * @return: Status of the operation.
     */
    static ReturnStatus get_rtp_param(const uint8_t* rtp_hdr, uint32_t& sequence_number,
                               uint32_t& timestamp, bool& m_bit, bool& f_bit);
    /**
     * @brief: Video stream packet sequence parsing status.
     */
    enum class RxState {
        syncing,
        receiving,
        paused
    };
    /**
     * @brief: Calculates the start time of the next video frame.
     *
     * @param [in] last_pkt_ts: Rx timestamp of the last packet of the current frame.
     *
     * @return: Timestamp of the next frame (N*Tframe).
     */
    uint64_t calc_next_frame_start(uint64_t last_pkt_ts);
    /**
     * @brief: Prepares a chunk for filling in the reply data.
     *
     * @param [out] chunk: Chunk handler to prepare a chunk.
     *
     * @return: Status of the operation.
     */
    ReturnStatus prepare_reply_chunk(std::shared_ptr<GenericChunk>& chunk);
    /**
     * @brief: Fills in the reply structure with latency measurements.
     *
     * @param [out] commit_chunk: Chunk to fill in.
     * @param [in] stats: Latency measurements.
     * @param [in] dropped_pkt_cnt: Number of dropped packets.
     *
     */
    void compose_reply(std::shared_ptr<GenericChunk> commit_chunk, const LatencyStats& stats,
                       uint64_t dropped_pkt_cnt);
};

} // io_node
} // dev_kit
} // rivermax

#endif /* RDK_IO_NODE_MISC_MEDIA_LATENCY_IO_NODE_H_ */
