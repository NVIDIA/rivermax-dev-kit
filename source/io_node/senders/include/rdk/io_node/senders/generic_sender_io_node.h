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

#ifndef RDK_IO_NODE_SENDERS_GENERIC_SENDER_IO_NODE_H_
#define RDK_IO_NODE_SENDERS_GENERIC_SENDER_IO_NODE_H_

#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>

#include <rivermax_api.h>

#include "rdk/io_node/common/checksum_buffer_writer.h"
#include "rdk/io_node/common/generic_buffer_writer.h"
#include "rdk/core/stream/send/generic_stream.h"
#include "rdk/core/flow/flow.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
{
namespace io_node
{
/**
 * @brief: Application Generic API send stream interface.
 *
 * This class implements and extends @ref GenericSendStream operations.
 */
class AppGenericSendStream : public GenericSendStream
{
private:
    std::vector<TwoTupleFlow> m_send_flows;
    size_t m_next_flow_to_send;
    rmx_mem_region m_mem_region;
public:
    /**
     * @brief: AppGenericSendStream constructor.
     *
     * @param [in] settings: Stream parameters.
     * @param [in] send_flows: Send flows the stream will use.
     */
    AppGenericSendStream(const GenericStreamSettings& settings, const std::vector<TwoTupleFlow>& send_flows);
    virtual ~AppGenericSendStream() = default;
    /**
     * @brief: Assignes a memory region to be used by the stream.
     *
     * @param [in] mreg: Memory region.
     */
    virtual void assign_mem_region(const rmx_mem_region& mreg);
    virtual ReturnStatus create_stream() override;
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
    MediaSettings m_media_settings;
    std::vector<std::unique_ptr<AppGenericSendStream>> m_streams;
    size_t m_index;
    size_t m_num_of_streams;
    int m_sleep_between_operations_us;
    bool m_print_parameters;
    FourTupleFlow m_network_address;
    PacketPacingRate m_rate;
    size_t m_num_of_chunks;
    size_t m_num_of_packets_in_chunk;
    uint16_t m_packet_typical_payload_size;
    uint16_t m_packet_typical_app_header_size;
    bool m_use_checksum_header;
    rmx_mem_region m_mem_region;
    int m_cpu_core_affinity;
    uint32_t m_hw_queue_full_sleep_us;
    std::shared_ptr<MemoryUtils> m_mem_utils;
    std::unique_ptr<IChunkBufferWriter> m_buffer_writer;
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
     * @brief: Overrides operator << for @ref GenericSenderIONode reference.
     */
    friend std::ostream& operator<<(std::ostream& out, const GenericSenderIONode& sender)
    {
        sender.print(out);
        return out;
    }
    /**
     * @brief: Overrides operator << for @ref GenericSenderIONode pointer.
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
     * using @ref GenericSenderIONode::distribute_memory_for_streams.
     *
     * @param [in] pointer: Pointer to the allocated memory for the chunks, this memory should be registered first.
     * @param [in] mkey: Memory key from @ref rmx_deregister_memory in Rivermax API.
     *
     * @return: The initialized memory length.
     */
    size_t initialize_memory(void* pointer, rmx_mkey_id mkey);
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
     *        @ref app_settings->print_parameters parameter of set to true.
     */
    void print_parameters();
    /**
     * @brief: Sender's worker.
     *
     * This method is the worker method of the std::thread will run with this object as it's context.
     * The user of @ref GenericSenderIONode class can
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
     * It should be called after @ref GenericSenderIONode::initialize_streams.
     */
    inline void prepare_buffers();
};

} // io_node
} // dev_kit
} // rivermax

#endif /* RDK_IO_NODE_SENDERS_GENERIC_SENDER_IO_NODE_H_ */
