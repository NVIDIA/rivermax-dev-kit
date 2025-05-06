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

#ifndef RDK_CORE_CHUNK_IPO_RECEIVE_CHUNK_H_
#define RDK_CORE_CHUNK_IPO_RECEIVE_CHUNK_H_

#include <unordered_set>

#include <rivermax_api.h>

#include "rdk/services/utils/defs.h"
#include "rdk/core/chunk/receive_chunk.h"

namespace rivermax
{
namespace dev_kit
{
namespace core
{

/**
 * @brief: This class represents virtual chunks of the aggregated receive stream.
 *
 * These chunks contain packets produced by @ref IPOReceiveStream
 * by combining packets from underlying physical redundant receive streams.
 */
class IPOReceiveChunk : public ReceiveChunk
{
    size_t m_chunk_size;
    uint32_t m_seqn_first;
    std::unordered_set<rmx_input_completion_flag> m_comp_flags;
    uint64_t m_timestamp_first;
    uint64_t m_timestamp_last;
    byte_t* m_header_ptr;
    byte_t* m_payload_ptr;
    const ReceivePacketInfo* m_info_ptr;
public:
    IPOReceiveChunk(bool use_hds) :
        ReceiveChunk(0, use_hds),
        m_chunk_size(0),
        m_seqn_first(0),
        m_timestamp_first(0),
        m_timestamp_last(0),
        m_header_ptr(nullptr),
        m_payload_ptr(nullptr),
        m_info_ptr(nullptr)
    {}

    ~IPOReceiveChunk() = default;

    size_t get_length() const override { return m_chunk_size; }
    const void* get_header_ptr() const override { return m_header_ptr; };
    const void* get_payload_ptr() const override { return m_payload_ptr; };
    const ReceivePacketInfo get_packet_info(size_t packet_idx) const override { return m_info_ptr[packet_idx]; }
    /**
     * @brief: Sets the number of packets in the chunk.
     *
     * @param [in] chunk_size: Number of packets in the chunk.
     */
    void set_completion_chunk_size(size_t chunk_size) { m_chunk_size = chunk_size; }
    /**
     * @brief: Returns the sequence number of the first packet in the chunk.
     *
     * @return: Sequence number of the first packet in the chunk.
     */
    uint32_t get_completion_seqn_first() const { return m_seqn_first; }
    /**
     * @brief: Sets the sequence number of the first packet in the chunk.
     *
     * @param [in] seqn: Sequence number of the first packet in the chunk.
     */
    void set_completion_seqn_first(uint32_t seqn) { m_seqn_first = seqn; }
    /**
     * @brief: Tests the status of the completion flag @p flag.
     *
     * @param [in] flag: Flag type, see @ref rmx_input_completion_flag.
     *
     * @return: True is flag is set.
     */
    bool test_completion_flag(rmx_input_completion_flag flag) const { return m_comp_flags.count(flag) != 0; }
    /**
     * @brief: Sets the status of the completion flag @p flag.
     *
     * @param [in] flag: Flag type, see @ref rmx_input_completion_flag.
     */
    void set_completion_flag(rmx_input_completion_flag flag) { m_comp_flags.insert(flag); }
    /**
     * @brief: Returns the packet header size.
     *
     * @param [in] packet_idx: Packet index in the chunk.
     *
     * @return: Packet header size.
     */
    size_t get_packet_header_size(size_t packet_idx) const { return is_header_data_split_on() ? m_info_ptr[packet_idx].get_packet_sub_block_size(0) : 0; }
    /**
     * @brief: Returns the packet payload size.
     *
     * @param [in] packet_idx: Packet index in the chunk.
     *
     * @return: Packet payload size.
     */
    size_t get_packet_payload_size(size_t packet_idx) const { return m_info_ptr[packet_idx].get_packet_sub_block_size(m_data_block_idx); }
    /**
     * @brief: Returns the timestamp of the specified paket in the chunk.
     *
     * @param [in] packet_idx: Packet index in the chunk.
     *
     * @return: Timestamp of the specified paket in the chunk.
     */
    uint64_t get_packet_timestamp(size_t packet_idx) const { return m_info_ptr[packet_idx].get_packet_timestamp(); }
    /**
     * @brief: Returns the first packet timestamp.
     *
     * @return: First packet timestamp.
     */
    uint64_t get_completion_timestamp_first() const { return m_timestamp_first; }
    /**
     * @brief: Sets the first packet timestamp.
     *
     * @param [in] timestamp: First packet timestamp.
     */
    void set_completion_timestamp_first(uint64_t timestamp) { m_timestamp_first = timestamp; }
    /**
     * @brief: Returns the last packet timestamp.
     *
     * @return: Last packet timestamp.
     */
    uint64_t get_completion_timestamp_last() const { return m_timestamp_last; }
    /**
     * @brief: Sets the last packet timestamp.
     *
     * @param [in] timestamp: Last packet timestamp.
     */
    void set_completion_timestamp_last(uint64_t timestamp) { m_timestamp_last = timestamp; }
    /**
     * @brief: Sets the pointer to the underlying header completion structure.
     *
     * @param [in] header_ptr: Pointer to the underlying header completion structure.
     */
    void set_completion_header_ptr(byte_t* header_ptr) { m_header_ptr = header_ptr; }
    /**
     * @brief: Sets the pointer to the underlying payload completion structure.
     *
     * @param [in] payload_ptr: Pointer to the underlying payload completion structure.
     */
    void set_completion_payload_ptr(byte_t* payload_ptr) { m_payload_ptr = payload_ptr; }
    /**
     * @brief: Returns the pointer to the array of packet info completion structures.
     *
     * @return: Pointer to the array of packet info completion structures.
     */
    const ReceivePacketInfo* get_completion_info_ptr() const {return m_info_ptr; }
    /**
     * @brief: Sets the pointer to the array of packet info completion structures.
     *
     * @param [in] info_ptr: Ppointer to the array of packet info completion structures.
     */
    void set_completion_info_ptr(const ReceivePacketInfo* info_ptr) { m_info_ptr = info_ptr; }
};

} // namespace core
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_CORE_CHUNK_IPO_RECEIVE_CHUNK_H_ */
