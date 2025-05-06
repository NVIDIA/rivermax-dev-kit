/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef RMAX_APPS_LIB_LIB_CORE_CHUNK_RECEIVE_CHUNK_H_

#include <cstddef>
#include <string>

#include <rivermax_api.h>

#include "core/chunk/chunk_interface.h"
#include "services/error_handling/return_status.h"

using namespace ral::lib::services;

namespace ral
{
namespace lib
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
class ReceiveChunk : public IChunk
{
private:
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
    virtual size_t get_length() const override { return rmx_input_get_completion_chunk_size(m_rmax_comp); }
    /**
     * @brief: Returns Rivermax internal completion.
     *
     * @return: Pointer to the underlay Rivermax completion.
     */
    const rmx_input_completion* get_completion() { return m_rmax_comp; }
    /**
     * @brief: Returns Rivermax internal header pointer.
     *
     * @return: Pointer to the underlay Rivermax header array.
     */
    const void* get_header_ptr() const { return m_hds_on ? rmx_input_get_completion_ptr(m_rmax_comp, 0) : nullptr; }
    /**
     * @brief: Returns Rivermax internal payload pointer.
     *
     * @return: Pointer to the underlay Rivermax payload array.
     */
    const void* get_payload_ptr() const { return rmx_input_get_completion_ptr(m_rmax_comp, m_data_block_idx); }
    /**
     * @brief: Returns the metainfo structure of the specified packet.
     *
     * @param [in] packet_idx: packet index in the chunk.
     *
     * @return: Metainfo structure of the specified packet, see @ref rmx_input_packet_info.
     */
    const ReceivePacketInfo get_packet_info(size_t packet_idx)
    {
        return ReceivePacketInfo(rmx_input_get_packet_info(&m_chunk_handle, packet_idx));
    }
    /**
     * @brief: Receives next chunk from input stream.
     *
     * @return: Status code as defined by @ref ReturnStatus.
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::signal_received - If operation was interrupted by an OS signal.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus get_next_chunk();
    /**
     * @brief: Returns status of Header-Data-Split mode.
     *
     * @return: true if Header-Data-Split mode is enabled.
     */
    bool is_hds_on() { return m_hds_on; }
};

} // namespace core
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_CORE_CHUNK_RECEIVE_CHUNK_H_
#endif /* RMAX_APPS_LIB_LIB_CORE_CHUNK_RECEIVE_CHUNK_H_ */
