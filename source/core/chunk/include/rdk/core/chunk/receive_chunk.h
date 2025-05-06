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

#ifndef RDK_CORE_CHUNK_RECEIVE_CHUNK_H_
#define RDK_CORE_CHUNK_RECEIVE_CHUNK_H_

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
 * @brief: This class provides access to received packet information.
 */
class ReceivePacketInfo {
public:
    /**
     * ReceivePacketInfo constructor.
     */
    ReceivePacketInfo(const rmx_input_packet_info* descr = nullptr) : m_descr(descr) {}
    /**
     * @brief: Returns the size of a packet sub-block.
     *
     * @param [in] sub_block_idx: Index of packet data sub-block.
     *
     * @return: The size of the contents of the specified packet and block ID, see @ref rmx_input_packet_info.
     */
    size_t get_packet_sub_block_size(size_t sub_block_idx) const
    {
        return rmx_input_get_packet_size(m_descr, sub_block_idx);
    }
    /**
     * @brief: Returns the user-assigned flow-tag associated with the specified packet.
     *
     * @return: User-assigned flow-tag associated with the specified packet, see @ref rmx_input_packet_info.
     */
    uint32_t get_packet_flow_tag() const
    {
        return rmx_input_get_packet_flow_tag(m_descr);
    }
    /**
     * @brief: Returns the arrival timestamp of the specified packet.
     *
     * @return: The arrival timestamp of the specified packet, see @ref rmx_input_packet_info.
     */
    uint64_t get_packet_timestamp() const
    {
        return rmx_input_get_packet_timestamp(m_descr);
    }
private:
    /**
     * @brief: Rivermax opaque packet info structure.
     */
    const rmx_input_packet_info* m_descr;
};

/**
 * @brief: RX API chunk interface class.
 */
class ReceiveChunk : public IReceiveChunk
{
protected:
    rmx_input_chunk_handle m_chunk_handle;
    const rmx_input_completion* m_rmax_comp;
    bool m_hds_on;
    size_t m_data_block_idx;
    rmx_stream_id m_stream_id;
public:
    /**
     * @brief: ReceiveChunk default constructor.
     *
     * @param [in] stream_id: ID of the owner stream.
     * @param [in] use_hds: Separate memory for packet headers.
     */
    ReceiveChunk(rmx_stream_id id, bool use_hds);
    size_t get_length() const override { return rmx_input_get_completion_chunk_size(m_rmax_comp); }
    const void* get_header_ptr() const override { return m_hds_on ? rmx_input_get_completion_ptr(m_rmax_comp, 0) : nullptr; }
    const void* get_payload_ptr() const override { return rmx_input_get_completion_ptr(m_rmax_comp, m_data_block_idx); }
    bool is_header_data_split_on() const override { return m_hds_on; }
    /**
     * @brief: Returns Rivermax internal completion.
     *
     * @return: Pointer to the underlay Rivermax completion.
     */
    const rmx_input_completion* get_completion() const { return m_rmax_comp; }
    /**
     * @brief: Returns the metainfo structure of the specified packet.
     *
     * @param [in] packet_idx: packet index in the chunk.
     *
     * @return: Metainfo structure of the specified packet, see @ref rmx_input_packet_info.
     */
    virtual const ReceivePacketInfo get_packet_info(size_t packet_idx) const
    {
        auto chunk_handle = m_chunk_handle;
        return ReceivePacketInfo(rmx_input_get_packet_info(&chunk_handle, packet_idx));
    }
    /**
     * @brief: Receives next chunk from input stream.
     *
     * @return: Status code as defined by @ref ReturnStatus.
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::signal_received - If operation was interrupted by an OS signal.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus get_next_chunk();
};

} // namespace core
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_CORE_CHUNK_RECEIVE_CHUNK_H_ */
