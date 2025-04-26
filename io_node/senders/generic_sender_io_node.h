/*
 * Copyright © 2017-2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef RMAX_APPS_LIB_IO_NODE_SENDERS_GENERIC_SENDER_IO_NODE_H_

#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>

#include <rivermax_api.h>

#include "api/rmax_apps_lib_api.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

namespace ral
{
namespace io_node
{

/**
 * @brief: Application Generic API send stream interface.
 *
 * This class implements and extends @ref ral::lib::core::GenericSendStream operations.
 */
class AppGenericSendStream : public GenericSendStream
{
private:
    std::vector<TwoTupleFlow> m_send_flows;
    size_t m_next_flow_to_send;
public:
    /**
     * @brief: AppGenericSendStream constructor.
     *
     * @param [in] network_address: Network address of the stream.
     * @param [in] send_flows: Send flows the stream will use.
     * @param [in] pp_rate: Packet pacing rate for the stream.
     *                      If packet pacing is not needed, the struct should be initialized to 0.
     * @param [in] num_of_requested_chunks: Number of chunks to be used in the stream.
     * @param [in] num_of_packets_in_chunk: Number of packets in chunk.
     * @param [in] packet_typical_payload_size: Packet typical payload size in bytes.
     * @param [in] packet_typical_app_header_size: Packet typical application header size in bytes.
     */
    AppGenericSendStream(
        const FourTupleFlow& network_address, std::vector<TwoTupleFlow>& send_flows,
        pp_rate_t rate, size_t num_of_requested_chunks, size_t num_of_packets_in_chunk,
        uint16_t packet_typical_payload_size, uint16_t packet_typical_app_header_size);
    virtual ~AppGenericSendStream() = default;
    std::ostream& print(std::ostream& out) const override;
    /**
     * @brief: Returns next destination flow to use, following application logic.
     *
     * @returns: Pointer to the next destination flow.
     */
    TwoTupleFlow* get_next_flow_to_send();
    /**
     * @brief: Sets next destination flow to use, following application logic.
     */
    void set_next_flow_to_send();
};
/**
 * @brief: GenericSenderIONode class.
 *
 * This class implements the required operations in order
 * to be a sender. The sender class will be the context
 * that will be run under a std::thread by overriding the operator ().
 * Each sender will be able to run multiple streams.
 */
class GenericSenderIONode
{
private:
    std::vector<std::unique_ptr<AppGenericSendStream>> m_streams;
    size_t m_index;
    size_t m_num_of_streams;
    int m_sleep_between_operations_us;
    bool m_print_parameters;
    FourTupleFlow m_network_address;
    pp_rate_t m_rate;
    size_t m_num_of_chunks;
    size_t m_num_of_packets_in_chunk;
    uint16_t m_packet_typical_payload_size;
    uint16_t m_packet_typical_app_header_size;
    bool m_use_checksum_header;
    gs_mem_block_t m_mem_block;
    rmax_cpu_set_t m_cpu_affinity_mask;
    int m_cpu_core_affinity;
    uint32_t m_hw_queue_full_sleep_us;
    std::unique_ptr<IBufferWriter> m_buffer_writer;
    std::shared_ptr<MemoryUtils> m_mem_utils;
    std::unordered_map<size_t,std::vector<TwoTupleFlow>> m_flows_in_stream;
public:
    /**
     * @brief: GenericSenderIONode constructor.
     *
     * @param [in] app_settings: The application settings.
     * @param [in] index: The index of the sender.
     * @param [in] num_of_streams: Number of streams in the sender.
     * @param [in] cpu_core_affinity: CPU core affinity the sender will run on.
     * @param [in] mem_utils: Memory utilities.
     */
    GenericSenderIONode(
        std::shared_ptr<AppSettings> app_settings,
        size_t index, size_t num_of_streams, int cpu_core_affinity,
        std::shared_ptr<MemoryUtils> mem_utils);
    virtual ~GenericSenderIONode() = default;
    /**
     * @brief: Returns sender's streams container.
     *
     * @return: Sender's streams container.
     */
    std::vector<std::unique_ptr<AppGenericSendStream>>& get_streams() { return m_streams; };
    /**
     * @brief: Prints sender's parameters to a output stream.
     *
     * The method prints the parameters of the sender to be shown to the user to a output stream.
     *
     * @param [out] out: Output stream parameter print to.
     *
     * @return: Reference to the output stream.
     */
    std::ostream& print(std::ostream& out) const;
    /**
     * @brief: Overrides operator << for @ref ral::io_node::GenericSenderIONode reference.
     */
    friend std::ostream& operator<<(std::ostream& out, const GenericSenderIONode& sender)
    {
        sender.print(out);
        return out;
    }
    /**
     * @brief: Overrides operator << for @ref ral::io_node::GenericSenderIONode pointer.
     */
    friend std::ostream& operator<<(std::ostream& out, GenericSenderIONode* sender)
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
     *
     * @param [in/out] flows_offset: The offset of the send flows in the upper application.
     *                               The parameter used to initialize network address of the streams.
     */
    void initialize_streams(size_t& flows_offset);
    /**
     * @brief: Initializes sender's memory.
     *
     * This method is responsible to take the memory assigned to the sender
     * and distribute it to the streams of this sender
     * using @ref ral::io_node::GenericSenderIONode::distribute_memory_for_streams.
     *
     * @param [in] pointer: Pointer to the allocated memory for the chunks, this memory should be registered first.
     * @param [in] mkey: Memory key from @ref rmax_deregister_memory in Rivermax API.
     *
     * @return: The initialized memory length.
     */
    size_t initialize_memory(void* pointer, rmax_mkey_id mkey);
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
     * The user of @ref ral::io_node::GenericSenderIONode class can
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
     * @brief: Destroy sender's streams.
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
     * It should be called after @ref ral::io_node::GenericSenderIONode::initialize_streams.
     */
    inline void prepare_buffers();
};

} // io_node
} // ral

#define RMAX_APPS_LIB_IO_NODE_SENDERS_GENERIC_SENDER_IO_NODE_H_
#endif /* RMAX_APPS_LIB_IO_NODE_SENDERS_GENERIC_SENDER_IO_NODE_H_ */
