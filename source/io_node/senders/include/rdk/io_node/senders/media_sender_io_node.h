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

#ifndef RDK_IO_NODE_SENDERS_MEDIA_SENDER_IO_NODE_H_
#define RDK_IO_NODE_SENDERS_MEDIA_SENDER_IO_NODE_H_

#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>

#include <rivermax_api.h>

#include "rdk/io_node/common/io_node_memory_utils.h"
#include "rdk/io_node/common/rtp_video_send_stream.h"
#include "rdk/io_node/common/chunk_buffer_writer_interface.h"
#include "rdk/services/media/media_frame_provider.h"
#include "rdk/services/buffer_wr/rtp_video_buffer_writer.h"
#include "rdk/core/memory_layout/header_payload_memory_layout.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
{
namespace io_node
{
/**
 * @brief: A timestamp for sending chunks immediately after pending chunks in the send queue.
 *
 * This timestamp is used to indicate the commit APIs to send chunks immediately
 * after pending chunks in the send queue, without waiting.
 */
constexpr uint64_t SEND_IMMEDIATELY_AFTER_PENDING_CHUNKS_TIMESTAMP = 0;
/**
 * @brief: MediaSenderIONode class.
 *
 * This class implements the required operations in order
 * to be a sender. The sender class will be the context
 * that will be run under a std::thread by overriding the operator ().
 * Each sender will be able to run multiple streams.
 */
class MediaSenderIONode : public IHeaderPayloadMemoryLayoutComponent
{
private:
    /**
    * @brief: Application media send stream resources.
    */
    struct MediaStreamPack
    {
        std::unique_ptr<RtpVideoSendStream> stream;
        std::unique_ptr<MediaChunk> chunk_handler;
        std::unique_ptr<MediaStreamMemBlockset> mem_blockset;
        std::vector<TwoTupleFlow> flows;
        std::unique_ptr<RTPMediaBufferWriter> buffer_writer;
        std::shared_ptr<IFrameProvider> frame_provider;
    };
    static constexpr size_t DEFAULT_NUMBER_OF_MEM_BLOCKS = 10;
    static constexpr size_t DEFAULT_PRINT_TIME_INTERVAL_MS = 1000;
    std::vector<MediaStreamPack> m_stream_packs;
    MediaSettings m_media_settings;
    std::string m_video_file;
    size_t m_index;
    FourTupleFlow m_network_address;
    int m_sleep_between_operations;
    bool m_print_parameters;
    int m_cpu_core_affinity;
    uint32_t m_hw_queue_full_sleep_us;
    IONodeMemoryUtils& m_memory_utils;
    size_t m_num_of_memory_blocks;
    size_t m_num_of_chunks_in_mem_block;
    uint16_t m_packet_header_size;
    uint16_t m_packet_payload_size;
    size_t m_num_of_packets_in_chunk;
    size_t m_num_of_packets_in_mem_block;
    size_t m_app_header_stride_size;
    size_t m_data_stride_size;
    size_t m_header_total_memory_size;
    size_t m_payload_total_memory_size;
    size_t m_block_payload_memory_size;
    size_t m_block_header_memory_size;
    std::vector<uint16_t> m_mem_block_header_sizes;
    std::vector<uint16_t> m_mem_block_payload_sizes;
    uint8_t m_dscp, m_pcp, m_ecn;
    time_handler_ns_cb_t m_get_time_ns_cb;
    bool m_gpu_enabled;
    bool m_dynamic_video_file_load;
    std::chrono::milliseconds m_print_interval_ms = std::chrono::milliseconds(DEFAULT_PRINT_TIME_INTERVAL_MS);
public:
    /**
     * @brief: MediaSenderIONode constructor.
     *
     * @param [in] network_address: Network address of the IO node.
     * @param [in] app_settings: Application settings.
     * @param [in] index: Index of the sender.
     * @param [in] num_of_streams: Number of streams in the sender.
     * @param [in] cpu_core_affinity: CPU core affinity the sender will run on.
     * @param [in] memory_utils: Memory utilities.
     * @param [in] time_hanlder_cb: Time handle callback the IO node will use to get current time.
     */
    MediaSenderIONode(
        const FourTupleFlow& network_address,
        std::shared_ptr<AppSettings> app_settings,
        size_t index, size_t num_of_streams, int cpu_core_affinity,
        IONodeMemoryUtils& memory_utils,
        time_handler_ns_cb_t time_hanlder_cb);
    virtual ~MediaSenderIONode() = default;
    ReturnStatus initialize_memory_layout() override;
    ReturnStatus determine_memory_layout(HeaderPayloadMemoryLayoutRequest& memory_layout_request) const override;
    ReturnStatus apply_memory_layout(const HeaderPayloadMemoryLayoutResponse& memory_layout_response) override;
    ReturnStatus validate_memory_layout(const HeaderPayloadMemoryLayoutResponse& memory_layout_respose) const override;
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
     * @brief: Overrides operator << for @ref MediaSenderIONode reference.
     */
    friend std::ostream& operator<<(std::ostream& out, const MediaSenderIONode& sender)
    {
        sender.print(out);
        return out;
    }
    /**
     * @brief: Overrides operator << for @ref MediaSenderIONode pointer.
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
     * @brief: Returns sender index.
     *
     * @return: Sender index.
     */
    virtual size_t get_index() const { return m_index; }
    /**
     * @brief: Sender's worker.
     *
     * This method is the worker method of the std::thread will run with this object as it's context.
     * The user of @ref MediaSenderIONode class can
     * initialize the object in advance and run std::thread when needed.
     */
    void operator()();
    /**
     * @brief: Sets the frame provider for the specified stream index.
     *
     * @param [in] stream_index: Stream index.
     * @param [in] frame_provider: Frame provider to set.
     * @param [in] media_type: Media type.
     * @param [in] contains_payload: Flag indicating whether the frame provider contains payload.
     *
     * @return: Status of the operation.
     */
    ReturnStatus set_frame_provider(size_t stream_index, std::shared_ptr<IFrameProvider> frame_provider,
        MediaType media_type = MediaType::Video, bool contains_payload = true);
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
     * It should be called after @ref MediaSenderIONode::initialize_streams.
     */
    inline void prepare_buffers();
    /**
     * @brief: Returns current time in nanoseconds.
     *
     * @returns: Current time in nanoseconds.
     */
    uint64_t get_time_now_ns() const { return m_get_time_ns_cb(nullptr); }
    /**
     * @brief: Waits for the next frame.
     *
     * This method implements logic to wait and wake up when next frame send time is close.
     *
     * @param [in] send_time_ns: Send time of the next frame in nanoseconds.
     */
    inline void wait_for_next_frame(uint64_t send_time_ns);
    /**
     * @brief: Returns status of Header-Data-Split mode.
     *
     * @return: true if Header-Data-Split mode is enabled.
     */
    bool is_hds_on() const { return m_packet_header_size != 0; }
    /**
     * @brief: Applies memory layout to subcomponents (streams) for Rivermax internal allocation.
     *
     * This method is helper method for  @ref apply_memory_layout()
     * and is responsible to distribute the memory of the sender to it's subcomponents (streams).
     * Internal allocation is used when the application doesn't provide externally allocated memory.
     *
     * @return: Status of the operation.
     */
    ReturnStatus apply_memory_layout_to_subcomponents();
    /**
     * @brief: Applies memory layout to subcomponents (streams) for application allocation.
     *
     * This method is helper method for  @ref apply_memory_layout()
     * and is responsible to distribute the memory of the sender to it's subcomponents (streams).
     * Application allocation is used when the application provides externally allocated memory.
     *
     * @param [in] memory_layout: Memory layout of the IO Node.
     *
     * @return: Status of the operation.
     */
    ReturnStatus apply_memory_layout_to_subcomponents(const HeaderPayloadMemoryLayout& memory_layout);
    /**
     * @brief: Initializes memory blockset with application allocation.
     *
     * This method initializes the memory blockset with application allocation.
     *
     * @param [in] mem_blockset: Memory blockset to initialize.
     * @param [in] header_memory_ptr: Pointer to header memory.
     * @param [in] payload_memory_ptr: Pointer to payload memory.
     * @param [in] io_node_memory_layout: IO Node memory layout.
     * @param [in] video_file: Video file to read frames from.
     *
     * @return: Status of the operation.
     */
    ReturnStatus initialize_mem_blockset(
        MediaStreamMemBlockset& mem_blockset,
        uint8_t* header_memory_ptr, uint8_t* payload_memory_ptr,
        const HeaderPayloadMemoryLayout& io_node_memory_layout,
        const std::string& video_file);
    /**
     * @brief: Initializes memory blockset with application allocation.
     *
     * This method initializes the memory blockset with application allocation.
     *
     * @param [in] mem_blockset: Memory blockset to initialize.
     * @param [in] header_memory_ptr: Pointer to header memory.
     * @param [in] payload_memory_ptr: Pointer to payload memory.
     * @param [in] io_node_memory_layout: IO Node memory layout.
     *
     * @return: Status of the operation.
     */
    ReturnStatus initialize_mem_blockset(
        MediaStreamMemBlockset& mem_blockset,
        uint8_t* header_memory_ptr, uint8_t* payload_memory_ptr,
        const HeaderPayloadMemoryLayout& io_node_memory_layout);
    /**
     * @brief: Initializes memory blockset with Rivermax allocation.
     *
     * This method initializes the memory blockset with Rivermax allocation.
     *
     * @param [in] mem_blockset: Memory blockset to initialize.
     *
     * @return: Status of the operation.
     */
    ReturnStatus initialize_mem_blockset(MediaStreamMemBlockset& mem_blockset);
    /**
     * @brief: Returns the memory requirements for a single block.
     *
     * This method calculates the memory requirements for a single block.
     *
     * @param [out] block_header_memory_size: Block header memory size.
     * @param [out] block_payload_memory_size: Block payload memory size.
     */
    void determine_memory_layout_for_single_block(size_t& block_header_memory_size, size_t& block_payload_memory_size);
    /**
     * @brief: Checks if internal allocation is requested.
     *
     * @param [in] layout: General memory layout details.
     *
     * @return: True if internal allocation is requested, false otherwise.
     */
    bool is_internal_allocation_requested(const HeaderPayloadMemoryLayout& layout) const;
    /**
     * @brief: Processes a frame.
     *
     * This method processes a frame by retrieving it from the frame provider and setting it in the buffer writer.
     *
     * @return: Status of the operation.
     */
    ReturnStatus process_frame();
    /**
     * @brief: Fills a memory block from a file.
     *
     * This method reads data from the specified input file and fills the provided memory block buffer.
     *
     * @param [out] block_memory_buffer: Pointer to the memory block buffer to fill.
     * @param [in] block_memory_size: Size of the memory block buffer.
     * @param [in] header_offset: Offset for the header in each payload stride.
     * @param [in] input_file: Input file stream to read data from.
     *
     * @return: Status of the operation.
     */
    ReturnStatus fill_memblock_from_file(byte_t* block_memory_buffer, size_t block_memory_size,
        size_t header_offset, std::ifstream& input_file) const;
    /**
     * @brief: Returns the number of memory blocks required for a file.
     *
     * This method calculates the number of memory blocks required to store the data from the specified file.
     *
     * @param [out] num_of_memory_blocks: Number of memory blocks required.
     *
     * @return: Status of the operation.
     */
    ReturnStatus get_number_of_mem_blocks_per_file(size_t& num_of_memory_blocks) const;
    /**
     * @brief: Returns the commit timestamp in nanoseconds.
     *
     * This method calculates the commit timestamp based on the provided parameters.
     *
     * @param [in] first_chunk_in_frame: Flag indicating if this is the first chunk in the frame.
     * @param [in] send_time_ns: Send time in nanoseconds.
     * @param [in] stream_id: ID of the stream.
     *
     * @return: Commit timestamp in nanoseconds.
     */
     inline uint64_t get_commit_timestamp_ns(bool first_chunk_in_frame, double send_time_ns, size_t stream_id) const;
};

inline uint64_t MediaSenderIONode::get_commit_timestamp_ns(
    bool first_chunk_in_frame, double send_time_ns, size_t stream_id) const {
    uint64_t current_time_ns = get_time_now_ns();

    if (first_chunk_in_frame && likely(send_time_ns > current_time_ns)) {
        return static_cast<uint64_t>(send_time_ns);
    } else if (unlikely(send_time_ns <= current_time_ns)) {
        static auto start_time = std::chrono::high_resolution_clock::now();
        auto time_now = std::chrono::high_resolution_clock::now();
        auto time_elapsed = time_now - start_time;
        if (time_elapsed >= m_print_interval_ms) {
            start_time = time_now;
            int timeout_ns = current_time_ns - send_time_ns;
            std::cout << "Sender " << m_index << ", Stream " << stream_id
                      << ": Timeout occurred. Send time exceeded by " << timeout_ns << " [ns]." << std::endl;
        }
    }

    return SEND_IMMEDIATELY_AFTER_PENDING_CHUNKS_TIMESTAMP;
}

} // io_node
} // dev_kit
} // rivermax

#endif /* RDK_IO_NODE_SENDERS_MEDIA_SENDER_IO_NODE_H_ */
