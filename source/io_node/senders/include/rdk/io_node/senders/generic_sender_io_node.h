/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <unordered_map>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>

#include <rivermax_api.h>

#include "rdk/io_node/io_node_interface.h"
#include "rdk/io_node/io_node_settings.h"
#include "rdk/io_node/common/checksum_buffer_writer.h"
#include "rdk/io_node/common/generic_buffer_writer.h"
#include "rdk/core/stream/send/generic_stream.h"
#include "rdk/core/flow/flow.h"
#include "rdk/services/cpu/affinity/rivermax_affinity.h"

using namespace rdk::services;
using namespace rdk::core;

namespace rdk
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
 * @brief: Settings of a generic sender I/O node.
 */
struct GenericSenderIONodeSettings : IONodeSettings
{
    /* Application settings the node was built from. */
    std::shared_ptr<const AppSettings> app_settings;
    /* Number of send streams the node owns. */
    size_t num_of_streams = 0;
    /* CPU core the node runs on, or @ref NO_CPU_AFFINITY. */
    int cpu_core_affinity = NO_CPU_AFFINITY;
    /* Send flows of each stream, keyed by the stream's index within the node. Filled by
     * @ref GenericSenderIONode::initialize_send_flows, so it is empty before the app calls it. */
    std::unordered_map<size_t, std::vector<TwoTupleFlow>> flows_in_stream;

    ReturnStatus validate() const override
    {
        if (IONodeSettings::validate() != ReturnStatus::success || !app_settings) {
            return ReturnStatus::failure;
        }
        return ReturnStatus::success;
    }
};

/**
 * @brief: GenericSenderIONode class.
 *
 * Implements the required operations to be a sender. Each sender can run multiple
 * streams.
 */
class GenericSenderIONode : public IIONode<GenericSenderIONodeSettings>
{
private:
    std::vector<std::unique_ptr<AppGenericSendStream>> m_streams;
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
    uint32_t m_hw_queue_full_sleep_us;
    std::shared_ptr<MemoryUtils> m_mem_utils;
    std::unique_ptr<IChunkBufferWriter> m_buffer_writer;
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
     * @brief: Requests the sender to stop, releasing the streams it holds.
     */
    void stop() noexcept override;
private:
    /**
     * @brief: Sender's worker: setup, send loop, teardown.
     */
    ReturnStatus do_work() override;
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

} // namespace io_node
} // namespace rdk

#endif /* RDK_IO_NODE_SENDERS_GENERIC_SENDER_IO_NODE_H_ */
