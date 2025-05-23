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

#ifndef RDK_CORE_STREAM_RECEIVE_RECEIVE_STREAM_H_
#define RDK_CORE_STREAM_RECEIVE_RECEIVE_STREAM_H_

#include <unordered_set>

#include <rivermax_api.h>

#include "rdk/services/error_handling/return_status.h"
#include "rdk/core/memory_layout/header_payload_memory_layout.h"
#include "rdk/core/stream/receive/single_receive_stream_interface.h"
#include "rdk/core/chunk/receive_chunk.h"
#include "rdk/core/flow/receive_flow.h"

namespace rivermax
{
namespace dev_kit
{
namespace core
{

/**
 * @brief: Settings for creating an input stream
 *
 * This class stores the stream configuration parameters and also
 * implements a builder that builds the stream descriptor structure
 * for creating a stream with Rivermax API.
 */
class ReceiveStreamSettings : public IStreamSettings<ReceiveStreamSettings, rmx_input_stream_params> {
public:
    /**
     * @brief: ReceiveStreamSettings constructor.
     *
     * @param [in] local_addr: NIC address to receive the stream.
     * @param [in] rx_type: Receive stream type, see @ref rmx_input_stream_params_type.
     * @param [in] ts_format: Timestamp format, see @ref rmx_input_timestamp_format.
     * @param [in] options: A set of stream options, see @ref rmx_input_option.
     * @param [in] capacity_in_packets: Number of packets in receive memory.
     * @param [in] payload_size: Packet payload size in bytes.
     * @param [in] header_size: Packet application header size in bytes
     *             (value > 0 specifies that a separate memory is used for headers).
     */
    ReceiveStreamSettings(const TwoTupleFlow& local_addr,
        rmx_input_stream_params_type rx_type,
        rmx_input_timestamp_format ts_format,
        const std::unordered_set<rmx_input_option>& options,
        size_t capacity_in_packets, size_t payload_size, size_t header_size);

    TwoTupleFlow m_local_addr;
    rmx_input_stream_params_type m_rx_type;
    rmx_input_timestamp_format m_ts_format;
    std::unordered_set<rmx_input_option> m_options;
    size_t m_capacity_in_packets;
    size_t m_payload_size;
    size_t m_header_size;
protected:
    /**
     * @brief: Initializes the input stream descriptor structure.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    void stream_param_init(rmx_input_stream_params& descr);
    /**
     * @brief: Sets stream local address.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    void stream_param_set_nic_address(rmx_input_stream_params& descr);
    /**
     * @brief: Sets number of packets in memory.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    void stream_param_set_capacity(rmx_input_stream_params& descr);
    /**
     * @brief: Sets stream packet size.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    void stream_param_set_packet_size(rmx_input_stream_params& descr);
    /**
     * @brief: Sets timestamp format.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    void stream_param_set_ts_format(rmx_input_stream_params& descr);
    /**
     * @brief: Sets stream options.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    void stream_param_set_input_options(rmx_input_stream_params& descr);
    /**
     * @brief: Sequence of parameter setters invoked to build
     *         an input stream descriptor structure.
     */
    static SetterSequence s_build_steps;
};

/**
 * @brief: Base RX stream class.
 *
 * This class implements @ref ISingleStream operations.
 * Wraps up Rivermax stream object and flow.
 */
class ReceiveStream: public ISingleReceiveStream
{
public:
    /**
     * @brief: Receive stream constructor.
     *
     * @param [in] settings: Receive stream settings structure, see @ref ReceiveStreamSettings.
     */
    ReceiveStream(const ReceiveStreamSettings& settings);
    /**
     * @brief: Destroys receive stream.
     */
    virtual ~ReceiveStream() = default;
    std::ostream& print(std::ostream& out) const override;
    ReturnStatus create_stream() override;
    ReturnStatus get_next_chunk(ReceiveChunk& chunk) override;
    ReturnStatus detach_flows() override;
    ReturnStatus destroy_stream() override;
    ReturnStatus initialize_memory_layout() override;
    ReturnStatus determine_memory_layout(HeaderPayloadMemoryLayoutRequest& memory_layout_request) const override;
    ReturnStatus apply_memory_layout(const HeaderPayloadMemoryLayoutResponse& memory_layout_response) override;
    ReturnStatus validate_memory_layout(const HeaderPayloadMemoryLayoutResponse& memory_layout_respose) const override;
    size_t get_header_stride_size() const override { return m_header_stride_size; }
    size_t get_payload_stride_size() const override { return m_data_stride_size; }
    bool is_header_data_split_on() const override { return m_stream_settings.m_header_size != 0; }
    /**
     * @brief: Configures a rule, how many packets to receive, or how much time to wait before
     *         returning the next requested chunk.
     *
     * @param [in] min_count: A minimal number of packets to return.
     * @param [in] max_count: A maximal number of packets to return.
     * @param [in] timeout_usec: A timeout in usec to wait for @p min_count of packets.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus set_completion_moderation(size_t min_count, size_t max_count, int timeout_usec);
    /**
     * @brief: Attaches a flow to the stream.
     *
     * @param [in] flow: Flow to attach to.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus attach_flow(ReceiveFlow& flow);
    /**
     * @brief: Detaches a flow from the stream.
     *
     * @param [in] flow: Flow to detach from.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus detach_flow(ReceiveFlow& flow);
    /**
     * @brief: Returns the memory size needed by Rivermax to create stream with given parameters.
     *
     * @note: TODO: should be removed after refactor.
     *        The function is invoked from Latency IO Node. After refactor it should be removed.
     *
     * @param [out] header_buffer_size: Header buffer size (if header-data split is enabled).
     * @param [out] payload_buffer_size: Payload buffer size.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus query_buffer_size(size_t& header_buffer_size, size_t& payload_buffer_size) const;
    /**
     * @brief: Sets buffers for header and payload data.
     *
     * @note: TODO: move to protected.
     *        The function is invoked from Latency IO Node. After refactor it should move to protected.
     *
     * @param [in] header_ptr: Pointer to header data buffer. Should be NULL if
     *                         header-data split mode is not enabled.
     * @param [in] payload_ptr: Pointer to payload data buffer.
     */
    void set_buffers(void* header_ptr, void* payload_ptr);
    /**
     * @brief: Sets memory regions for header and payload memory.
     *
     * @note: TODO: move to protected.
     *        The function is invoked from Latency IO Node. After refactor it should move to protected.
     *
     * @param [in] header_mkey: Memory key for header data buffer. Ignored if
     *                          header-data split mode is not enabled.
     * @param [in] payload_mkey: Memory key for payload data buffer.
     */
    void set_memory_keys(rmx_mkey_id header_mkey, rmx_mkey_id payload_mkey);
protected:
    /**
     * @brief: Returns the memory size needed by Rivermax to create stream with given parameters.
     *
     * @param [out] header_buffer_size: Header buffer size (if header-data split is enabled).
     * @param [out] payload_buffer_size: Payload buffer size.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus determine_memory_layout_helper(size_t& header_buffer_size, size_t& payload_buffer_size) const;
protected:
    ReceiveStreamSettings m_stream_settings;
    rmx_input_stream_params m_stream_params;
    /* These parameters are calculated in query_buffer_size() */
    size_t m_data_stride_size;
    size_t m_header_stride_size;
    /* This parameter is set in constructor, but is updated if query_buffer_size() is called */
    size_t m_buffer_elements;
    const size_t m_header_mem_block_id;
    const size_t m_payload_mem_block_id;
    std::unordered_set<ReceiveFlow> m_flows;
    rmx_mem_region* m_header_block;
    rmx_mem_region* m_payload_block;
};

} // namespace core
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_CORE_STREAM_RECEIVE_RECEIVE_STREAM_H_ */
