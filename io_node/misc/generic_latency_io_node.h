/*
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef RMAX_APPS_LIB_IO_NODE_MISC_GENERIC_LATENCY_IO_NODE_H_
#define RMAX_APPS_LIB_IO_NODE_MISC_GENERIC_LATENCY_IO_NODE_H_

#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>

#include <rivermax_api.h>
#include "latency_io_node.h"

#include <api/rmax_apps_lib_api.h>

using namespace ral::lib::core;
using namespace ral::lib::services;

namespace ral
{
namespace io_node
{

/**
 * @brief: Data dimensions of a generic stream.
 */
struct StreamDimensions {
    size_t num_of_chunks;
    size_t num_of_packets_in_chunk;
    size_t header_size;
    size_t payload_size;
    StreamDimensions(size_t a_num_of_chunks, size_t a_num_of_packets,
                     size_t a_header_size, size_t a_payload_size) :
        num_of_chunks(a_num_of_chunks),
        num_of_packets_in_chunk(a_num_of_packets),
        header_size(a_header_size),
        payload_size(a_payload_size) {};
};

/**
 * @brief: GenericLatencyIONode class.
 *
 * This class implements resource control operations for latency measurement modes
 * based on Generic API. In particular, it is responsible for initialization of
 * Generic Output and Receive streams, used for transmit and receive of data packets
 * during latency measurement.
 */
class GenericLatencyIONode: public LatencyIONode
{
protected:
    std::shared_ptr<GenericSendStream> m_send_stream;
    std::shared_ptr<ReceiveStream> m_receive_stream;
    StreamDimensions m_send_dim;
    StreamDimensions m_receive_dim;
    uint32_t m_hw_queue_full_sleep_us;
    bool m_track_completions;
    std::unique_ptr<IBufferWriter> m_buffer_writer;
public:
    /**
     * @brief: GenericLatencyIONode constructor.
     *
     * @param [in] settings: Latency measurement settings.
     * @param [in] send_dim: Numerical parameters of the send stream.
     * @param [in] receive_dim: Numerical parameters of the receive stream.
     * @param [in] header_mem_utils: Header Memory utilities.
     * @param [in] payload_mem_utils: Payload Memory utilities.
     * @param [in] time_handler_cb: Timer handler.
     */
    GenericLatencyIONode(
        const LatencyNodeSettings& settings,
        const StreamDimensions& send_dim,
        const StreamDimensions& receive_dim,
        std::shared_ptr<MemoryUtils> header_mem_utils,
        std::shared_ptr<MemoryUtils> payload_mem_utils,
        time_handler_ns_cb_t time_handler_cb);
    virtual ~GenericLatencyIONode() = default;
    virtual std::ostream& print(std::ostream& out) const;
    /**
     * @brief: Overrides operator << for @ref ral::io_node::GenericLatencyIONode reference.
     */
    friend std::ostream& operator<<(std::ostream& out, const GenericLatencyIONode& sender)
    {
        sender.print(out);
        return out;
    }
    /**
     * @brief: Overrides operator << for @ref ral::io_node::GenericLatencyIONode pointer.
     */
    friend std::ostream& operator<<(std::ostream& out, GenericLatencyIONode* sender)
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
    /**
     * @brief: Returns true, if headers are sent from a separate memory, otherwise false.
     */
    bool is_send_hds() const { return m_send_dim.header_size != 0; }
    /**
     * @brief: Returns true, if headers are received into a separate memory, otherwise false.
     */
    bool is_receive_hds() const { return m_receive_dim.header_size != 0; }
protected:
    ReturnStatus create_send_stream() override;
    ReturnStatus destroy_send_stream() override;
    ReturnStatus create_receive_stream() override;
    ReturnStatus destroy_receive_stream() override;
    ReturnStatus attach_receive_flow() override;
    virtual ReturnStatus detach_receive_flow() override;
};

/**
 * @brief: PingPongIONode class.
 *
 * This class implements Ping-Pong latency measurement mode.
 * In this mode a single packet is sent from Client to Server and immediately sent backwards,
 * and round-trip time is measured.
 */
class PingPongIONode: public GenericLatencyIONode
{
public:
    /**
     * @brief: PingPongIONode constructor.
     *
     * @param [in] settings: Latency measurement settings.
     * @param [in] header_mem_utils: Header Memory utilities.
     * @param [in] payload_mem_utils: Payload Memory utilities.
     * @param [in] time_handler_cb: Timer handler.
     */
    PingPongIONode(
        const LatencyNodeSettings& settings,
        std::shared_ptr<MemoryUtils> header_mem_utils,
        std::shared_ptr<MemoryUtils> payload_mem_utils,
        time_handler_ns_cb_t time_handler_cb);
    virtual ~PingPongIONode() {};
    /**
     * @brief: Overrides operator << for @ref ral::io_node::PingPongNode reference.
     */
    friend std::ostream& operator<<(std::ostream& out, const PingPongIONode& sender)
    {
        sender.print(out);
        return out;
    }
    /**
     * @brief: Overrides operator << for @ref ral::io_node::PingPongNode pointer.
     */
    friend std::ostream& operator<<(std::ostream& out, PingPongIONode* sender)
    {
        sender->print(out);
        return out;
    }
    void prepare_send_buffer() override;
protected:
    void send_receive() override;
    void receive_send() override;
};

/**
 * @brief: FrameIONode class.
 *
 * This class implements Frame latency measurement mode.
 * In this mode a data array (represenfing a video frame) is senf from Client ot Server
 * with maximal speed, and the time required for the whole operation is measured.
 */
class FrameIONode: public GenericLatencyIONode
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
    FrameIONode(
        const LatencyNodeSettings& settings,
        std::shared_ptr<MemoryUtils> header_mem_utils,
        std::shared_ptr<MemoryUtils> payload_mem_utils,
        time_handler_ns_cb_t time_handler_cb);
    virtual ~FrameIONode() {};
    /**
     * @brief: Overrides operator << for @ref ral::io_node::FrameIONode reference.
     */
    friend std::ostream& operator<<(std::ostream& out, const FrameIONode& sender)
    {
        sender.print(out);
        return out;
    }
    /**
     * @brief: Overrides operator << for @ref ral::io_node::FrameIONode pointer.
     */
    friend std::ostream& operator<<(std::ostream& out, FrameIONode* sender)
    {
        sender->print(out);
        return out;
    }
protected:
#pragma pack(push, 1)
    /*
     * packet_header and receive_timing structures are transmitted over the network as is,
     * without any byte ordering (BE/LE) conversion. This needs to be addressed in hypothetic
     * case of porting Rivermax to a Big-endian architecture.
     */
    struct packet_header {
        uint32_t iter_num;
        uint32_t pkt_num;
    };
    struct receive_timing {
        uint64_t first_rx_ts;
        uint64_t last_rx_ts;
        uint64_t soft_rx_ts;
        uint32_t iter_num;
        uint32_t valid;
    };
#pragma pack(pop)
    void prepare_send_buffer() override;
    void send_receive() override;
    void receive_send() override;

    uint32_t m_rx_next_pkt_num;
    uint32_t m_rx_iteration;
    size_t m_rx_drop_cnt;

private:
    /**
     * @brief: Prepares frame data for sending: embeds counters in each packet.
     */
    void prepare_chunks(uint32_t iteration);
    /**
     * @brief: Extracts receive timestamps from the server reply packets.
     *
     * @return: True if the packet is correct, otherwise false.
     */
    bool parse_receive_timing(ReceiveChunk& chunk, receive_timing& timing);
    /**
     * @brief: Parses a received chunk of packets from the client's frame.
     *
     * @param [in] receive_chunk: Chunk of packets to process.
     * @param [out] has_last_packet: THis chunk contains the last packet of the current frame.
     * @param [in] valid: True if the frame has no missing packets and a correct number,
     *                    otherwise false.

     * @return: True if the packet is correct, otherwise false.
     */
    void parse_packet_headers(ReceiveChunk& receive_chunk, bool& has_last_packet,
                              bool& valid);
    ReturnStatus prepare_reply_chunk(std::shared_ptr<GenericChunk>& chunk);
    void compose_reply(std::shared_ptr<GenericChunk> commit_chunk, uint64_t first_packet_ts,
                       uint64_t last_packet_ts, uint64_t receive_ts,
                       uint32_t iteration, bool valid);
};

} // io_node
} // ral

#endif /* RMAX_APPS_LIB_IO_NODE_MISC_GENERIC_LATENCY_IO_NODE_H_ */
