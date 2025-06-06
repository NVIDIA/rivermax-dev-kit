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

#ifndef RDK_CORE_CHUNK_MEDIA_CHUNK_H_
#define RDK_CORE_CHUNK_MEDIA_CHUNK_H_

#include <cstddef>

#include <rivermax_api.h>

#include "rdk/core/chunk/chunk_interface.h"
#include "rdk/services/error_handling/return_status.h"

using namespace rivermax::dev_kit::services;

namespace rivermax
{
namespace dev_kit
{
namespace core
{

/**
 * @brief: Media API chunk interface class.
 *
 * This interfaces indented to wrap Rivermax Media API chunk.
 */
class MediaChunk : public IChunk
{
private:
    rmx_output_media_chunk_handle m_chunk;
    void* m_data_ptr;
    size_t m_length;
    bool m_hds_on;
    rmx_stream_id m_stream_id;
public:
    /**
     * @brief: MediaChunk default constructor.
     *
     * @param [in] stream_id: ID of the owner stream.
     * @param [in] packets_in_chunk: Number of packets in chunk.
     * @param [in] use_hds: Separate memory for packet headers.
     */
    MediaChunk(rmx_stream_id id, size_t packets_in_chunk, bool use_hds);
    virtual size_t get_length() const override { return m_length; }
    /**
     * @brief: Returns a pointer to the underlay payload array of the chunk.
     *
     * @returns: Pointer to the data sizes array.
     */
    void* get_data_ptr();
    /**
     * @brief: Returns a pointer to the underlay application header array of the chunk.
     *
     * @returns: Pointer to the application header array.
     */
    void* get_app_hdr_ptr();
    /**
     * @brief: Returns a pointer to the underlay payload sizes array of the chunk.
     *
     * @returns: Pointer to the application header sizes array.
     */
    uint16_t* get_data_sizes_array();
    /**
     * @brief: Returns a pointer to the underlay application header sizes array of the chunk.
     *
     * @returns: Pointer to the application header sizes array.
     */
    uint16_t* get_app_hdr_sizes_array();
    /**
     * @brief: Acquires the next free chunk for a media output stream.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ReturnStatus::no_free_chunks - In case of insufficient available chunks.
     */
    ReturnStatus get_next_chunk();
    /**
     * @brief: Enables a chunk commit options. The option affect only the next commit.
     *
     * @param [in] options: Commit options to enable.
     */
    void set_commit_option(rmx_output_commit_option option);
    /**
     * @brief: Disables a chunk commit options. The option affect only the next commit.
     *
     * @param [in] options: Commit options to disable.
     */
    void clear_commit_option(rmx_output_commit_option option);
    /**
     * @brief: Disables all commit options. The options affect only the next commit.
     */
    void clear_all_commit_options();
    /**
     * @brief: Sends a chunk of the associated media output stream to the wire.
     *
     * @param [in] time: Time to schedule chunk sending, the format depends on the options
     * set in @ref set_commit_options.
     *
     * @warinig: If multiple chunks were acquired with @ref get_next_chunk for the
     * stream after the previous call to @ref commit_chunk, the oldest acquired chunk will
     * be sent, not the one whose method was called.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ReturnStatus::hw_send_queue_full - In case the HW send queue is full.
     *          @ref ReturnStatus::signal_received - In case a signal was received during the operation.
     */
    ReturnStatus commit_chunk(uint64_t time);
    /**
     * @brief: Return to Rivermax all chunks that were allocated with get_next_chunk,
     * but not committed.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ReturnStatus::hw_send_queue_full - In case the HW send queue is full.
     *          @ref ReturnStatus::signal_received - In case a signal was received during the operation.
     */
    ReturnStatus cancel_unsent();
    /**
     * @brief: Marks the chunk for completion tracking.
     *
     * @param [out] token: Arbitrary user token associated with the chunk.
     *
     * @warinig: To mark a chunk, call this function after @ref get_next_chunk
     * but before committing it with @ref commit_chunk.
     *
     * @return: Status code as defined by @ref rmx_status.
     */
    ReturnStatus mark_for_tracking(uint64_t token);
    /**
     * @brief: Polls for a new transmit completion of a chunk marked for completion tracking.
     *
     * @warning: When only some of the committed chunks were marked for tracking, this function may return
     *          @ref ReturnStatus::no_completion even if the marked chunks are already transmitted,
     *          but their status is not yet fetched from HW due to many preceeding non-marked completions.
     *          In this use case the function must be called multiple times until ReturnStatus::success
     *          is returned. Then the marked completion info can be read with @ref get_last_completion_info.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::no_completion - In case when no new completions are available.
     */
    ReturnStatus poll_for_completion();
    /**
     * @brief: Retrieves information of the last Tx completion obtained with @ref poll_for_completion.
     *
     * @param [out] timestamp: Time of completing the chunk transmit set by HW.
     * @param [out] token: Token associated with the chunk when marking for trackinw with @ref mark_for_tracking.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of incorrect or obsolete stream.
     */
    ReturnStatus get_last_completion_info(uint64_t& timestamp, uint64_t& token);
    /**
     * @brief: Returns status of Header-Data-Split mode.
     *
     * @return: true if Header-Data-Split mode is enabled.
     */
    bool is_hds_on() const { return m_hds_on; }
};

} // namespace core
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_CORE_CHUNK_MEDIA_CHUNK_H_ */
