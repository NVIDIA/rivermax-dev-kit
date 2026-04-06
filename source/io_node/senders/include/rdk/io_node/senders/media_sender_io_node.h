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

#ifndef RDK_IO_NODE_SENDERS_MEDIA_SENDER_IO_NODE_H_
#define RDK_IO_NODE_SENDERS_MEDIA_SENDER_IO_NODE_H_

#include <cstddef>
#include <iostream>
#include <memory>
#include <ostream>
#include <rivermax_api.h>
#include <vector>

#include "rdk/core/memory_layout/header_payload_memory_layout.h"
#include "rdk/io_node/common/chunk_buffer_writer_interface.h"
#include "rdk/io_node/common/io_node_memory_utils.h"
#include "rdk/io_node/common/rtp_video_send_stream.h"
#include "rdk/services/media/media_essence_source.h"
#include "rdk/services/media/media_settings.h"
#include "rdk/services/ulp_packet_buffer/ulp_packet_buffer.h"
#include "rdk/services/utils/synchronizer.h"

using namespace rdk::services;
using namespace rdk::core;

namespace rdk
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
        std::unique_ptr<MediaSendStream> stream;
        std::unique_ptr<MediaChunk> chunk_handler;
        std::unique_ptr<MediaStreamMemBlockset> mem_blockset;
        std::vector<FourTupleFlow> flows;
        std::unique_ptr<IULPPacketBufferWriter> runtime_packet_buffer_writer;
        std::shared_ptr<IMediaEssenceSource> runtime_essence_source;

        // Preload configuration: Used to fill memory blocks before transmission starts.
        std::shared_ptr<IULPPacketBufferWriter> preload_packet_buffer_writer;
        std::shared_ptr<IMediaEssenceSource> preload_essence_source;
        size_t number_of_memory_blocks;
        uint8_t* header_memory_ptr = nullptr;
        uint8_t* payload_memory_ptr = nullptr;
    };
    static constexpr size_t DEFAULT_PRINT_TIME_INTERVAL_MS = 1000;
    std::vector<MediaStreamPack> m_stream_packs;
    const MediaSettings& m_media_settings;
    size_t m_index;
    size_t m_num_paths_per_stream;
    int m_sleep_between_operations;
    bool m_print_parameters;
    uint32_t m_stats_report_interval_ms;
    uint64_t m_stats_sent_media_unit_chunk_counter;
    int m_cpu_core_affinity;
    uint32_t m_hw_queue_full_sleep_us;
    IONodeMemoryUtils& m_memory_utils;
    size_t m_block_header_memory_size;
    size_t m_block_payload_memory_size;
    size_t m_header_total_memory_size;
    size_t m_payload_total_memory_size;
    std::vector<uint16_t> m_mem_block_header_sizes;
    std::vector<uint16_t> m_mem_block_payload_sizes;
    uint8_t m_dscp, m_pcp, m_ecn;
    time_handler_ns_cb_t m_get_time_ns_cb;
    bool m_gpu_enabled;
    std::shared_ptr<ISynchronizer> m_synchronizer;
    std::chrono::milliseconds m_print_interval_ms = std::chrono::milliseconds(DEFAULT_PRINT_TIME_INTERVAL_MS);
public:
    /**
     * @brief: MediaSenderIONode constructor.
     *
     * @param [in] num_paths_per_stream: Number of paths per stream (for 2022-7 duplication).
     * @param [in] app_settings: Application settings.
     * @param [in] media_settings: Media settings.
     * @param [in] index: Index of the sender.
     * @param [in] num_of_streams: Number of streams in the sender.
     * @param [in] cpu_core_affinity: CPU core affinity the sender will run on.
     * @param [in] memory_utils: Memory utilities.
     * @param [in] time_hanlder_cb: Time handle callback the IO node will use to get current time.
     */
    MediaSenderIONode(
        size_t num_paths_per_stream,
        const AppSettings& app_settings,
        const MediaSettings& media_settings,
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
    void initialize_send_flows(const std::vector<FourTupleFlow>& flows);
    /**
     * @brief: Initializes stream objects.
     *
     * @return: Status of the operation.
     */
    ReturnStatus initialize_streams();
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
     * @brief: Sets media essence sources for a specific stream.
     *
     * This method configures the media essence sources that supply media data to a stream.
     * Two types of sources can be configured:
     *
     * - **Preload Source**: Pre-fills memory blocks with media data before transmission begins.
     *   This is a one-time operation that prepares data in advance for optimal performance.
     *
     * - **Runtime Source**: Supplies fresh media data dynamically during active transmission.
     *   Called continuously as new media units are available.
     *
     * @par Usage Patterns:
     * 1. **Static Content**: Set only a preload source and disable runtime payload copying
     *    (`runtime_contains_payload = false`) for maximum efficiency when transmitting
     *    the same data repeatedly.
     *
     * 2. **Dynamic Content**: Set only a runtime source when media data changes continuously.
     *
     * 3. **Hybrid Mode**: Set both sources - preload fills memory blocks once before
     *    transmission starts, while runtime supplies new media units to send when they
     *    become available during the transmission loop.
     *
     * @par Default Behavior:
     * Each stream is initialized with @ref NullEssenceSource for both sources by default.
     * @ref NullEssenceSource generates only RTP headers; payload data is not written
     * during the transmission loop. At least one source should be set to a real
     * implementation for meaningful data transmission.
     *
     * @param [in] stream_index: The index of the stream to configure.
     * @param [in] smpte_standard: The SMPTE standard for media formatting.
     * @param [in] preload_essence_source: Source for preloading data into memory blocks
     *                                     before transmission. Pass nullptr to preserve the
     *                                     existing preload source (default: nullptr).
     * @param [in] runtime_essence_source: Source for supplying media data during active
     *                                     transmission. Pass nullptr to preserve the existing
     *                                     runtime source (default: nullptr).
     * @param [in] runtime_contains_payload: If true, copies both headers and payload from the
     *                                       runtime source. If false, only constructs RTP
     *                                       headers from the runtime source, leaving payload
     *                                       data untouched (assumes preloaded). Setting to false
     *                                       improves performance when payload is static
     *                                       (default: true).
     *
     * @note: The @p runtime_contains_payload parameter only affects the runtime source's behavior.
     *        The @p preload_essence_source always writes complete data (headers and payload).
     *
     * @return: Status of the operation.
     */
    ReturnStatus set_media_essence_sources(
        size_t stream_index,
        SMPTEStandard smpte_standard,
        std::shared_ptr<IMediaEssenceSource> preload_essence_source = nullptr,
        std::shared_ptr<IMediaEssenceSource> runtime_essence_source = nullptr,
        bool runtime_contains_payload = true);
    /**
     * @brief: Sets the synchronizer for the sender.
     *
     * @param [in] synchronizer: Synchronizer to set.
     */
    void set_synchronizer(const std::shared_ptr<ISynchronizer>& synchronizer) { m_synchronizer = synchronizer; }
    static constexpr size_t DEFAULT_NUMBER_OF_MEM_BLOCKS = 1;
protected:
    /**
     * @brief: Prints sender statistics.
     *
     * @param [out] out: Output stream to print statistics to.
     * @param [in] interval_duration: Statistics interval duration.
     */
    virtual void print_statistics(std::ostream& out,
        const std::chrono::high_resolution_clock::duration& interval_duration) const;
    /**
     * @brief: Resets statistics.
     */
    virtual void reset_statistics();
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
     * @brief: Preloads media data into memory blocks.
     *
     * This method pre-fills memory blocks with media data from preload essence sources
     * before transmission starts. It processes all media units for each memory block,
     * writing the data using preload packet buffer writers. This optimization allows
     * for efficient transmission by having data ready in memory blocks in advance.
     */
    inline void preload_media_data();
    /**
     * @brief: Returns current time in nanoseconds.
     *
     * @returns: Current time in nanoseconds.
     */
    uint64_t get_time_now_ns() const { return m_get_time_ns_cb(nullptr); }
    /**
     * @brief: Waits for the next media unit.
     *
     * This method implements logic to wait and wake up when next media unit send time is close.
     *
     * @param [in] send_time_ns: Send time of the next media unit in nanoseconds.
     */
    inline void wait_for_next_media_unit(uint64_t send_time_ns);
    /**
     * @brief: Returns status of Header-Data-Split mode.
     *
     * @return: true if Header-Data-Split mode is enabled.
     */
    bool is_hds_on() const { return m_media_settings.packet_app_header_size != 0; }
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
     * @param [in] number_of_memory_blocks: Number of memory blocks.
     *
     * @return: Status of the operation.
     */
    ReturnStatus initialize_mem_blockset(
        MediaStreamMemBlockset& mem_blockset,
        uint8_t* header_memory_ptr, uint8_t* payload_memory_ptr,
        const HeaderPayloadMemoryLayout& io_node_memory_layout,
        size_t number_of_memory_blocks);
    /**
     * @brief: Initializes memory blockset with Rivermax allocation.
     *
     * This method initializes the memory blockset with Rivermax allocation.
     *
     * @param [in] mem_blockset: Memory blockset to initialize.
     * @param [in] number_of_memory_blocks: Number of memory blocks.
     *
     * @return: Status of the operation.
     */
    ReturnStatus initialize_mem_blockset(MediaStreamMemBlockset& mem_blockset,
        size_t number_of_memory_blocks);
    /**
     * @brief: Checks if internal allocation is requested.
     *
     * @param [in] layout: General memory layout details.
     *
     * @return: True if internal allocation is requested, false otherwise.
     */
    bool is_internal_allocation_requested(const HeaderPayloadMemoryLayout& layout) const;
    /**
     * @brief: Processes a media essence unit.
     *
     * This method processes a media unit by retrieving it from the media essence source and setting it in the buffer writer.
     *
     * @return: Status of the operation.
     */
    ReturnStatus process_media_unit();
    /**
     * @brief: Returns the commit timestamp in nanoseconds.
     *
     * This method calculates the commit timestamp based on the provided parameters.
     *
     * @param [in] first_chunk_in_media_unit: Flag indicating if this is the first chunk in the media unit.
     * @param [in] send_time_ns: Send time in nanoseconds.
     * @param [in] stream_id: ID of the stream.
     *
     * @return: Commit timestamp in nanoseconds.
     */
    inline uint64_t get_commit_timestamp_ns(bool first_chunk_in_media_unit, double send_time_ns, size_t stream_id) const;
    /**
     * @brief: Coordinates the start time with the synchronizer if available.
     *
     * This method handles the synchronization logic to coordinate the start time
     * across multiple senders using the provided synchronizer.
     *
     * @param [in,out] send_time_ns: The proposed send time, which may be adjusted by the synchronizer.
     *
     * @return: Status of the operation.
     */
    ReturnStatus coordinate_start_time(uint64_t& send_time_ns);
    /**
     * @brief: Calculates the required number of memory blocks based on the essence size.
     *
     * @param [in] essence_size: Total size in bytes of the media essence.
     *
     * @return: Required number of memory blocks.
     */
    size_t calculate_required_memory_blocks(size_t essence_size) const;
};

inline uint64_t MediaSenderIONode::get_commit_timestamp_ns(
    bool first_chunk_in_media_unit, double send_time_ns, size_t stream_id) const {
    uint64_t current_time_ns = get_time_now_ns();

    if (first_chunk_in_media_unit && likely(send_time_ns > current_time_ns)) {
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

} // namespace io_node
} // namespace rdk

#endif /* RDK_IO_NODE_SENDERS_MEDIA_SENDER_IO_NODE_H_ */
