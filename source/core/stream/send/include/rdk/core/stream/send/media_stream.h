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

#ifndef RDK_CORE_STREAM_SEND_MEDIA_STREAM_H_
#define RDK_CORE_STREAM_SEND_MEDIA_STREAM_H_

#include <cstddef>
#include <ostream>
#include <vector>

#include <rivermax_api.h>

#include "rdk/services/error_handling/return_status.h"
#include "rdk/core/memory_layout/media_memory_layout.h"
#include "rdk/core/memory_layout/header_payload_memory_layout.h"
#include "rdk/services/media/media_defs.h"
#include "rdk/core/stream/send/send_stream_interface.h"
#include "rdk/core/chunk/media_chunk.h"
#include "rdk/core/flow/flow.h"

namespace rivermax
{
namespace dev_kit
{
namespace core
{

/**
 * @brief: Settings for creating a media output stream.
 *
 * This class stores the stream configuration parameters and also
 * implements a builder that builds the stream descriptor structure
 * for creating a stream with Rivermax API.
 */
class MediaStreamSettings : public IStreamSettings<MediaStreamSettings, rmx_output_media_stream_params> {
public:
    /**
     * @brief: MediaSendStream constructor.
     *
     * @param [in] local_address: Network address of the stream.
     * @param [in] media_settings: Parameters of SMPTE-2110 media.
     * @param [in] packets_per_chunk: Number of packets in each chunk.
     * @param [in] packet_payload_size: Packet payload size in bytes.
     * @param [in] data_stride_size: Number of bytes in Rivermax stride for data.
     * @param [in] app_header_stride_size: Number of bytes in Rivermax stride for headers.
     * @param [in] dscp: DSCP value.
     * @param [in] pcp: PCP value.
     * @param [in] ecn: ECN value.
     */
    MediaStreamSettings(const TwoTupleFlow& local_address, const MediaSettings& media_settings,
            size_t packets_per_chunk, uint16_t packet_payload_size,
            size_t data_stride_size, size_t app_header_stride_size = 0,
            uint8_t dscp = 0, uint8_t pcp = 0, uint8_t ecn = 0);
    virtual ~MediaStreamSettings() = default;
    TwoTupleFlow m_local_address;
    MediaSettings m_media_settings;
    size_t m_packets_per_chunk;
    uint16_t m_packet_payload_size;
    size_t m_data_stride_size;
    size_t m_app_header_stride_size;
    uint8_t m_dscp;
    uint8_t m_pcp;
    uint8_t m_ecn;
protected:
    /**
     * @brief: Initializes the media output stream descriptor structure.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    void stream_param_init(rmx_output_media_stream_params& descr);
    /**
     * @brief: Sets stream parameters from the SDP specification.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    void stream_param_set_sdp(rmx_output_media_stream_params& descr);
    /**
     * @brief: Sets number of packets in chunk.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    void stream_param_set_packets_per_chunk(rmx_output_media_stream_params& descr);
    /**
     * @brief: Sets data and (if HDS is on) header stride sizes.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    void stream_param_set_stride_sizes(rmx_output_media_stream_params& descr);
    /**
     * @brief: Sets stream frame size in packets.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    void stream_param_set_packets_per_frame(rmx_output_media_stream_params& descr);
    /**
     * @brief: Sets PCP.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    void stream_param_set_pcp(rmx_output_media_stream_params& descr);
    /**
     * @brief: Sets DSCP.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    void stream_param_set_dscp(rmx_output_media_stream_params& descr);
    /**
     * @brief: Sets ECN.
     *
     * @param [out] descr: Stream descriptor opaque structure.
     */
    void stream_param_set_ecn(rmx_output_media_stream_params& descr);
    /**
     * @brief: Sequence of parameter setters invoked to build
     *         a media output stream descriptor structure.
     */
    static SetterSequence s_build_steps;
};

/**
 * @brief: Memory layout for media output stream.
 *
 * This class defines the location of packet memory blocks. Memory for output packets
 * is defined as one or more memory block, each block containing one or many packets.
 * A memory block consists of one or two sub-blocks. If each packet is stored in
 * a single continous memory area, thre is only one sub-block. If in each packet
 * application header and payload are stored in separate memory areas, tow sub-blocks
 * are defined: one for headers, and the other for payloads.
 * A @ref MediaStreamMemBlockset object is used to configure and ininalize an array of
 * @ref rmx_output_media_mem_block opaque structures, that are used in construction
 * of a @ref MediaSendStream object.
 */
class MediaStreamMemBlockset {
private:
    std::vector<rmx_output_media_mem_block> m_blocks;
    size_t m_sub_block_count;
    size_t m_chunks_per_block;
    /**
     * @note: Sizes of strides and chunks are stream parameters
     *        and specified in @ref MediaStreamSettings class.
     */
    friend class MediaSendStream;
    /**
     * @brief: Returns memory blocks for media output stream.
     *
     * @return: Array of @ref rmx_output_media_mem_block structures for memory blocks
     * defined for the media output stream.
     */
    rmx_output_media_mem_block* get_memory_blocks() { return m_blocks.data(); };
public:
    /**
    * @brief: MediaStreamMemBlockset constructor.
    *
    * The MediaStreamMemBlockset class specifies layout of all the data memory used by
    * media output stream.
    *
    * @param [in] block_count: Number of memory blocks to use.
    * @param [in] sub_block_count:  Number of sub-blocks in block: 2 for separate header/payload sub-blocks, 1 otherwise.
    * @param [in] chunks_per_block: Number of chunks in block.
    */
    MediaStreamMemBlockset(size_t block_count, size_t sub_block_count, size_t chunks_per_block);
    /**
    * @brief: Returns number of memory blocks.
    *
    * @return: Number of memory blocks.
    */
    size_t get_memory_block_count() const { return m_blocks.size(); };
    /**
    * @brief: Returns number of chunks in block.
    *
    * @return: Number of chunks in block.
    */
    size_t get_chunks_per_block() const { return m_chunks_per_block; };
    /**
    * @brief: Sets the memory parameters for one memory block.
    *
    * @param [in] idx: Block number, see @ref MediaStreamMemBlockset constructor.
    * @param [in] sub_block_idx: Sub-block number, see @ref MediaStreamMemBlockset constructor.
    * @param [in] block_memory_start: Start address of the block memory.
    * @param [in] block_memory_size: Size of the block memory.
    * @param [in] memory_key: Memory key.
    */
    void set_block_memory(size_t idx, size_t sub_block_idx, void* block_memory_start,
            size_t block_memory_size, rmx_mkey_id memory_key);
    /**
    * @brief: Sets the memory parameters for one memory block with two memory keys.
    *
    * @param [in] idx: Block number, see @ref MediaStreamMemBlockset constructor.
    * @param [in] sub_block_idx: Sub-block number, see @ref MediaStreamMemBlockset constructor.
    * @param [in] block_memory_start: Start address of the block memory.
    * @param [in] block_memory_size: Size of the block memory.
    * @param [in] memory_keys: Memory keys array (@ref RMX_MAX_SUB_BLOCKS_PER_MEM_BLOCK elements).
    */
    void set_dup_block_memory(size_t idx, size_t sub_block_idx, void* block_memory_start,
            size_t block_memory_size, rmx_mkey_id memory_keys[]);
    /**
    * @brief: Configures all memory blocks to be allocated by Rivermax.
    */
    void set_rivermax_to_allocate_memory();
    /**
    * @brief: Sets the sizes for all packets in one block.
    *
    * @param [in] idx: Block number, see @ref MediaStreamMemBlockset constructor.
    * @param [in] data_sizes: Array of sizes for packets in the block (or payload sub-blocks if HDS is on).
    * @param [in] app_header_sizes: Array of sizes for application headers in the block
    *                                (should be NULL if HDS is off).
    *
    * @return: Status of the operation.
    */
    ReturnStatus set_block_layout(size_t idx, uint16_t data_sizes[], uint16_t app_header_sizes[]);
};

/**
 * @brief: Media API send stream interface.
 *
 * This class implements @ref ISendStream operations.
 * It uses Rivermax TX media API.
 */
class MediaSendStream : public ISendStream, public IMemoryLayoutComponent<HeaderPayloadMemoryLayoutRequest, MediaMemoryLayoutResponse>
{
protected:
    MediaStreamSettings m_stream_settings;
    rmx_output_media_stream_params m_stream_params;
public:
    /**
     * @brief: MediaSendStream constructor without pre-configured memory.
     *
     * Creates a MediaSendStream object with no asigned memory regions.
     * Before creating a Rivermax stream with @ref create_stream need to
     * assign memory blocks with @ref assign_memory_blocks.
     *
     * @param [in] settings: Stream parameters.
     */
    MediaSendStream(const MediaStreamSettings& settings);
    /**
     * @brief: MediaSendStream constructor with pre-configured memory.
     *
     * Creates a MediaSendStream object with asigned memory blocks.
     *
     * @param [in] settings: Stream parameters.
     * @param [in] mem_blocks: Parameters of blocks allocated for output packets.
     */
    MediaSendStream(const MediaStreamSettings& settings, MediaStreamMemBlockset& mem_blocks);
    /**
     * @brief: Assigns memory blocks to the stream.
     *
     * @note: TODO: move to protected.
     *        The function is invoked from Latency IO Node. After refactor it should move to protected.
     *
     * @param [in] mem_blocks: Parameters of blocks allocated for output packets.
     */
    void assign_memory_blocks(MediaStreamMemBlockset& mem_blocks);
    std::ostream& print(std::ostream& out) const override;
    virtual ReturnStatus create_stream() override;
    virtual ReturnStatus destroy_stream() override;
    ReturnStatus initialize_memory_layout() override;
    ReturnStatus determine_memory_layout(HeaderPayloadMemoryLayoutRequest& memory_layout_request) const override;
    ReturnStatus apply_memory_layout(const MediaMemoryLayoutResponse& memory_layout_response) override;
    ReturnStatus validate_memory_layout(const MediaMemoryLayoutResponse& memory_layout_respose) const override;
    /**
     * @brief: Returns data stride size of the stream buffer attributes.
     *
     * @return: Data stride size.
     */
    virtual size_t get_data_stride_size() const { return m_stream_settings.m_data_stride_size; }
    /**
     * @brief: Returns header stride size of the stream buffer attributes.
     *
     * @return: Header stride size.
     */
    virtual size_t get_app_header_stride_size() const { return m_stream_settings.m_app_header_stride_size; }
    /**
     * @brief: Returns status of Header-Data-Split mode.
     *
     * @return: true if Header-Data-Split mode is enabled.
     */
    bool is_hds_on() const { return m_stream_settings.m_app_header_stride_size != 0; }
    /**
     * @brief: Acquires the next free chunk of the stream.
     *
     * @param [out] chunk: Free chunk aquired from the stream.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ReturnStatus::no_free_chunks - In case of insufficient available chunks.
     */
    ReturnStatus get_next_chunk(MediaChunk& chunk);
    /**
     * @brief: Acquires the next free chunk of the stream, a blocking helper.
     *
     * This function acquires the next free chunk for a media output stream,
     * it makes @ref retries attempts to acquire a chunk with @ref get_next_chunk.
     *
     * @param [out] chunk: Free chunk aquired from the stream.
     * @param [in] retries: Number of attempts to acuire a chunk.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ReturnStatus::no_free_chunks - In case of insufficient available chunks.
     *          @ref ReturnStatus::signal_received - In case a signal was received during the operation.
     */
    ReturnStatus blocking_get_next_chunk(MediaChunk& chunk, size_t retries = BLOCKING_CHUNK_RETRIES);
    /**
     * @brief: Sends a chunk of the stream to the wire.
     *
     * @param [in] chunk: Chunk to commit.
     * @param [in] time: Time to schedule chunk sending, the format depends on the options
     * set in @ref MediaChunk::set_commit_options.
     *
     * @warinig: If multiple chunks were acquired with @ref get_next_chunk for the
     * stream after the previous call to @ref commit_chunk, the oldest acquired chunk will
     * be sent, not the one passes in @p chunk.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ReturnStatus::hw_send_queue_full - In case the HW send queue is full.
     *          @ref ReturnStatus::signal_received - In case a signal was received during the operation.
     */
    ReturnStatus commit_chunk(MediaChunk& chunk, uint64_t time);
    /**
     * @brief: Sends a chunk of the stream to the wire, a blocking helper.
     *
     * @param [in] chunk: Chunk to commit.
     * @param [in] time: Time to schedule chunk sending, the format depends on the options
     * set in @ref MediaChunk::set_commit_options.
     * @param [in] retries: Number of attempts to send a chunk.
     *
     * @warinig: If multiple chunks were acquired with @ref get_next_chunk for the
     * stream after the previous call to @ref commit_chunk, the oldest acquired chunk will
     * be sent, not the one passes in @p chunk.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ReturnStatus::hw_send_queue_full - In case the HW send queue is full.
     *          @ref ReturnStatus::signal_received - In case a signal was received during the operation.
     */
    ReturnStatus blocking_commit_chunk(MediaChunk& chunk, uint64_t time, size_t retries = BLOCKING_COMMIT_RETRIES);
    /**
     * @brief: Return to Rivermax all chunks that were allocated with get_next_chunk,
     * but not committed.
     *
     * @param [in] chunk: A chhunk acquired with with @ref get_next_chunk but not committed.

     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ReturnStatus::hw_send_queue_full - In case the HW send queue is full.
     *          @ref ReturnStatus::signal_received - In case a signal was received during the operation.
     */
    ReturnStatus cancel_unsent_chunks(MediaChunk& chunk);
};

} // namespace core
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_CORE_STREAM_SEND_MEDIA_STREAM_H_ */
