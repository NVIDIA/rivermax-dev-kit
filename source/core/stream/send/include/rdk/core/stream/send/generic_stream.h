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

#ifndef RDK_CORE_STREAM_SEND_GENERIC_STREAM_H_
#define RDK_CORE_STREAM_SEND_GENERIC_STREAM_H_

#include <cstddef>
#include <ostream>
#include <vector>

#include <rivermax_api.h>

#include "rdk/services/utils/defs.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/core/memory_layout/header_payload_memory_layout.h"
#include "rdk/core/stream/send/send_stream_interface.h"
#include "rdk/core/chunk/generic_chunk.h"
#include "rdk/core/flow/flow.h"

namespace rivermax
{
namespace dev_kit
{
namespace core
{

/**
 * @brief: Settings for creating a generic output stream.
 *
 * This class stores the stream configuration parameters and also
 * implements a builder that builds the stream descriptor structure
 * for creating a stream with Rivermax API.
 */
class GenericStreamSettings : public IStreamSettings<GenericStreamSettings, rmx_output_gen_stream_params> {
public:
    FourTupleFlow m_network_address;
    bool m_fixed_dest_addr;
    PacketPacingRate m_pp_rate;
    size_t m_num_of_requested_chunks;
    size_t m_num_of_packets_in_chunk;
    uint16_t m_packet_typical_payload_size;
    uint16_t m_packet_typical_app_header_size;
    /**
     * @brief: GenericStreamSettings constructor.
     *
     * @param [in] network_address: Network address of the stream.
     * @param [in] fixed_dest_addr: Use a fixed destination address for the stream.
     * @param [in] pp_rate: Packet pacing rate for the stream.
     *                      If packet pacing is not needed, the struct should be initialized to 0.
     * @param [in] num_of_requested_chunks: Number of chunks to be used in the stream.
     * @param [in] num_of_packets_in_chunk: Number of packets in each chunk.
     * @param [in] packet_typical_payload_size: Packet typical payload size in bytes.
     * @param [in] packet_typical_app_header_size: Packet typical application header size in bytes.
     */
    GenericStreamSettings(const FourTupleFlow& network_address, bool fixed_dest_addr,
        PacketPacingRate pp_rate, size_t num_of_requested_chunks, size_t num_of_packets_in_chunk,
        uint16_t packet_typical_payload_size, uint16_t packet_typical_app_header_size);
    virtual ~GenericStreamSettings() = default;
protected:
    /**
     * @brief: Initializes the generic output stream descriptor structure.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    virtual void stream_param_init(rmx_output_gen_stream_params& descr);
    /**
     * @brief: Sets stream local address.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    void stream_param_set_local_addr(rmx_output_gen_stream_params& descr);
    /**
     * @brief: Sets stream remote address.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    void stream_param_set_remote_addr(rmx_output_gen_stream_params& descr);
    /**
     * @brief: Sets stream rate paratemers.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    void stream_param_set_rate(rmx_output_gen_stream_params& descr);
    /**
     * @brief: Sets stream chunk parameters.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    void stream_param_set_chunk_size(rmx_output_gen_stream_params& descr);
    /**
     * @brief: Sequence of parameter setters invoked to build
     *         a generic output stream descriptor structure.
     */
    static SetterSequence s_build_steps;
};

/**
 * @brief: Generic API send stream interface.
 *
 * This class implements @ref ISendStream operations.
 * It uses Rivermax TX generic API.
 */
class GenericSendStream : public ISendStream, public IHeaderPayloadMemoryLayoutComponent
{
protected:
    GenericStreamSettings m_stream_settings;
    std::vector<std::shared_ptr<GenericChunk>> m_chunks;
    size_t m_next_chunk_to_send_index;
    rmx_output_gen_stream_params m_stream_params;
    size_t m_header_buffer_size = 0;
    size_t m_payload_buffer_size = 0;
public:
    /**
     * @brief: GenericSendStream constructor.
     *
     * @param [in] settings: Stream parameters.
     */
    GenericSendStream(const GenericStreamSettings& settings);
    std::ostream& print(std::ostream& out) const override;
    ReturnStatus create_stream() override;
    ReturnStatus destroy_stream() override;
    ReturnStatus initialize_memory_layout() override;
    ReturnStatus determine_memory_layout(HeaderPayloadMemoryLayoutRequest& memory_layout_request) const override;
    ReturnStatus apply_memory_layout(const HeaderPayloadMemoryLayoutResponse& memory_layout_response) override;
    ReturnStatus validate_memory_layout(const HeaderPayloadMemoryLayoutResponse& memory_layout_respose) const override;
    /**
     * @brief: Returns a generic chunk from the chunk pool by index.
     *
     * @note: The user can override this function in case
     *        different logic and data structures for chunk management needed.
     *
     * @param [in] index: Index of the chunk.
     *
     * @return: Pointer to the generic chunk.
     */
    virtual std::shared_ptr<GenericChunk>& get_chunk(size_t index) { return m_chunks[index]; };
    /**
     * @brief: Returns the first generic chunk in the chunk pool.
     *
     * @return: Pointer to the generic chunk.
     */
    std::shared_ptr<GenericChunk>& get_chunk() { return m_chunks[0]; };
    /**
     * @brief: Returns the next free generic chunk.
     *
     * @note: The user can override this function in case
     *        different logic and data structures for chunk management needed.
     *
     * @param [in] chunk: Reference to pointer to the returned chunk.
     *
     * @return: Status of the operation.
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ReturnStatus::no_free_chunks - In case of insufficient available chunks.
     *          @ref ReturnStatus::signal_received - In case a signal was received during the operation.
     */
    virtual ReturnStatus get_next_chunk(std::shared_ptr<GenericChunk>& chunk);
    /**
     * @brief: Acquires the next free chunk, a blocking helper.
     *
     * This function acquires the next free chunk from the stream,
     * it makes @p retries attempts to acquire a chunk with @ref get_next_chunk.
     *
     * @param [in] chunk: Reference to pointer to the returned chunk.
     * @param [in] retries: Number of attempts to acuire a chunk.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ReturnStatus::no_free_chunks - In case of insufficient available chunks.
     *          @ref ReturnStatus::signal_received - In case a signal was received during the operation.
     */
    virtual ReturnStatus blocking_get_next_chunk(std::shared_ptr<GenericChunk>& chunk, size_t retries = BLOCKING_CHUNK_RETRIES);
    /**
     * @brief: Initializes chunks layout using the specified memory region.
     *
     * @param [in] mem_region: - Memory region for packets data.
     *
     * @note: TODO: move to protected.
     *        The function is invoked from Latency IO Node. After refactor it should move to protected.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    virtual ReturnStatus initialize_chunks(const rmx_mem_region& mem_region);
    /**
     * @brief: Initializes chunks layout using the specified header and payload memory regions.
     *
     * @param [in] header_region: Memory region for packets application headers.
     * @param [in] payload_region: Memory region for packets payload.
     *
     * @note: TODO: move to protected.
     *        The function is invoked from Latency IO Node. After refactor it should move to protected.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    virtual ReturnStatus initialize_chunks(const rmx_mem_region& header_region, const rmx_mem_region& payload_region);
    /**
     * @brief: Returns memory length needed for the stream including packet headers and payload.
     *
     * @return: Size in bytes of the memory length.
     */
    virtual size_t get_memory_length() const;
    /**
     * @brief: Returns memory length needed for packet application headers.
     *
     * @return: Size in bytes of the memory length.
     */
    virtual size_t get_header_memory_length() const;
    /**
     * @brief: Returns memory length needed for packet payloads.
     *
     * @return: Size in bytes of the memory length.
     */
    virtual size_t get_payload_memory_length() const;
    /**
     * @brief: Sends a chunk of the stream to the wire.
     *
     * @param [in] chunk: Pointer to the chunk to commit.
     * @param [in] time: Time to schedule chunk sending, the format depends on the options
     * set in @ref set_commit_options.
     *
     * @warinig: If multiple chunks were acquired with @ref get_next_chunk for the
     * stream after the previous call to @ref commit_chunk, the oldest acquired chunk will
     * be sent, not the one passed as @p chunk.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ReturnStatus::hw_send_queue_full - In case the HW send queue is full.
     *          @ref ReturnStatus::signal_received - In case a signal was received during the operation.
     */
    ReturnStatus commit_chunk(std::shared_ptr<GenericChunk> chunk, uint64_t time);
    /**
     * @brief: Sends a chunk of the stream to the wire, a blocking helper.
     *
     * @param [in] chunk: Pointer to the chunk to commit.
     * @param [in] time: Time to schedule chunk sending, the format depends on the options
     * set in @ref set_commit_options.
     * @param [in] retries: Number of attempts to send a chunk.
     *
     * @warinig: If multiple chunks were acquired with @ref get_next_chunk for the
     * stream after the previous call to @ref commit_chunk, the oldest acquired chunk will
     * be sent, not the one passed as @p chunk.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ReturnStatus::hw_send_queue_full - In case the HW send queue is full.
     *          @ref ReturnStatus::signal_received - In case a signal was received during the operation.
     */
    ReturnStatus blocking_commit_chunk(std::shared_ptr<GenericChunk> chunk, uint64_t time, size_t retries = BLOCKING_COMMIT_RETRIES);
};

} // namespace core
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_CORE_STREAM_SEND_GENERIC_STREAM_H_ */
