/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef RMAX_APPS_LIB_IO_NODE_SENDERS_MEDIA_SENDER_IO_NODE_H_

#include <cstddef>
#include <vector>
#include <unordered_map>
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
 * @brief: MediaSenderIONode class.
 *
 * This class implements the required operations in order
 * to be a sender. The sender class will be the context
 * that will be run under a std::thread by overriding the operator ().
 * Each sender will be able to run multiple streams.
 */
class MediaSenderIONode
{
private:
    /**
    * @brief: Application media send stream rerources
    */
    struct MediaStreamPack
    {
        std::unique_ptr<RtpVideoSendStream> stream;
        std::unique_ptr<MediaChunk> chunk_handler;
        std::unique_ptr<MediaStreamMemBlockset> mem_blockset;
        std::vector<TwoTupleFlow> flows;
    };
    std::vector<MediaStreamPack> m_stream_packs;
    media_settings_t m_media_settings;
    size_t m_index;
    FourTupleFlow m_network_address;
    int m_sleep_between_operations;
    bool m_print_parameters;
    int m_cpu_core_affinity;
    uint32_t m_hw_queue_full_sleep_us;
    std::unique_ptr<IBufferWriter> m_buffer_writer;
    std::shared_ptr<MemoryUtils> m_mem_utils;
    size_t m_num_of_chunks_in_mem_block;
    uint16_t m_packet_payload_size;
    size_t m_num_of_packets_in_chunk;
    size_t m_num_of_packets_in_mem_block;
    size_t m_data_stride_size;
    std::vector<uint16_t> m_mem_block_payload_sizes;
    uint8_t m_dscp, m_pcp, m_ecn;
    time_handler_ns_cb_t m_get_time_ns_cb;
public:
    /**
     * @brief: MediaSenderIONode constructor.
     *
     * @param [in] network_address: Network address of the IO node.
     * @param [in] app_settings: Application settings.
     * @param [in] index: Index of the sender.
     * @param [in] num_of_streams: Number of streams in the sender.
     * @param [in] cpu_core_affinity: CPU core affinity the sender will run on.
     * @param [in] mem_utils: Memory utilities.
     * @param [in] time_hanlder_cb: Time handle callback the IO node will use to get current time.
     */
    MediaSenderIONode(
        const FourTupleFlow& network_address,
        std::shared_ptr<AppSettings> app_settings,
        size_t index, size_t num_of_streams, int cpu_core_affinity,
        std::shared_ptr<MemoryUtils> mem_utils, time_handler_ns_cb_t time_hanlder_cb);
    virtual ~MediaSenderIONode() = default;
    /**
     * @brief: Prints sender's parameters to a output stream.
     *
     * The method prints the parameters of the sender to be shown to the user to a output stream.
     *
     * @param [out] out: Output stream parameter print to.
     *
     * @return: Output stream.
     */
    std::ostream& print(std::ostream& out) const;
    /**
     * @brief: Overrides operator << for @ref ral::io_node::MediaSenderIONode reference.
     */
    friend std::ostream& operator<<(std::ostream& out, const MediaSenderIONode& sender)
    {
        sender.print(out);
        return out;
    }
    /**
     * @brief: Overrides operator << for @ref ral::io_node::MediaSenderIONode pointer.
     */
    friend std::ostream& operator<<(std::ostream& out, MediaSenderIONode* sender)
    {
        sender->print(out);
        return out;
    }
    /**
     * @brief: Initializes send flows objects.
     *
     * This method will initialize the send flows uniformly among it's streams.
     *
     * @param [in] flows: Flows assigned to sender's streams.
     */
    void initialize_send_flows(const std::vector<TwoTupleFlow>& flows);
    /**
     * @brief: Initializes stream objects.
     */
    void initialize_streams();
    /**
     * @brief: Initializes sender's memory.
     *
     * This method is responsible to take the memory assigned to the sender
     * and distribute it to the streams of this sender
     * using @ref ral::io_node::MediaSenderIONode::distribute_memory_for_streams.
     */
    void initialize_memory();
    /**
     * @brief: Distributes the memory for streams.
     *
     * This method is responsible to distribute the memory of the sender
     * to it's streams.
     */
    void distribute_memory_for_streams();
    /**
     * @brief: Prints sender's parameters.
     *
     * This method is responsible to use the @ref operators << overloaded
     * of sender and it's streams and print the information to standard output.
     *
     * @note: The information will be printed if the sender was initialized with
     * @ref app_settings->print_parameters parameter of set to true.
     */
    void print_parameters();
    /**
     * @brief: Sender's worker.
     *
     * This method is the worker method of the std::thread will run with this object as it's context.
     * The user of @ref ral::io_node::MediaSenderIONode class can
     * initialize the object in advance and run std::thread when needed.
     */
    void operator()();
private:
    /**
     * @brief: Creates sender's streams.
     *
     * This method is responsible to go over sender's stream objects and
     * create the appropriate Rivermax stream.
     *
     * @return: Status of the operation.
     */
    ReturnStatus create_streams();
    /**
     * @brief: Destroys sender's streams.
     *
     * This method is responsible to go over sender's stream objects and
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
    /**
     * @brief: Prepares the buffers to send.
     *
     * This method is responsible to prepare the data to be sent for it's streams.
     * It should be called after @ref ral::io_node::MediaSenderIONode::initialize_streams.
     */
    inline void prepare_buffers();
    /**
     * @brief: Returns current time in nanoseconds.
     *
     * @returns: Current time in nanoseconds.
     */
    uint64_t get_time_now_ns() { return m_get_time_ns_cb(nullptr); }
    /**
     * @brief: Waits for the next frame.
     *
     * This method implements logic to wait and wake up when next frame send time is close.
     *
     * @param [in] send_time_ns: Send time of the next frame in nanoseconds.
     */
    inline void wait_for_next_frame(uint64_t send_time_ns);
};

} // io_node
} // ral

#define RMAX_APPS_LIB_IO_NODE_SENDERS_MEDIA_SENDER_IO_NODE_H_
#endif /* RMAX_APPS_LIB_IO_NODE_SENDERS_MEDIA_SENDER_IO_NODE_H_ */
