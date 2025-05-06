/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef RMAX_APPS_LIB_LIB_CORE_CHUNK_GENERIC_CHUNK_H_

#include <cstddef>
#include <string>
#include <vector>
#include <unordered_set>

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
 * @brief: Generic output stream packet.
 *
 * The packet is defined by a vector of memory regions that it occupies.
 * The maximum number of memory regions in the packet is limited by @ref RMX_MAX_SUB_BLOCKS_PER_MEM_BLOCK.
 */
typedef std::vector<rmx_mem_region> GenericPacket;

/**
 * @brief: Generic API chunk interface class.
 *
 * This interfaces intended to wrap Rivermax Generic API chunk.
 */
class GenericChunk : public IChunk
{
private:
    rmx_output_gen_chunk_handle m_chunk;
    rmx_stream_id m_stream_id;
    std::vector<GenericPacket> m_packets;
public:
    /**
     * @brief: GenericChunk default constructor.
     *
     * @param [in] stream_id: ID of the owner stream.
     * @param [in] packets_in_chunk: Number of packets in chunk.
     */
    GenericChunk(rmx_stream_id stream_id, size_t packets_in_chunk);
    virtual size_t get_length() const override { return m_packets.size(); }
    /**
     * @brief: Returns Rivermax internal chunk.
     *
     * @returns: Pointer to the underlay Rivermax chunk.
     */
    rmx_output_gen_chunk_handle* get_rmax_chunk() { return& m_chunk; }
    /**
     * @brief: Place a packet into the chunk.
     *
     * This function sets the specified packet at position @p packet_idx in the chunk.
     *
     * @param [in] packet_idx: Packet index.
     * @param [in] packet: Packet to place into the chunk.
     *
     * @return: Status of the operation.
     */
    void place_packet(size_t packet_idx, const GenericPacket& packet) { m_packets[packet_idx] = packet; }
    /**
     * @brief: Returns a packet by its index.
     *
     * This function returns a reference to a chunk packet.
     * The reference can be used to fill in the packet data or to change packet location in memory.
     *
     * @param [in] packet_idx: Packet index.
     *
     * @return: The specified chunk packet.
     */
    GenericPacket& get_packet(size_t packet_idx) { return m_packets[packet_idx]; }
    /**
     * @brief: Acquire the next free chunk for a generic output stream.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ral::lib::services::ReturnStatus::no_free_chunks - In case of insufficient available chunks.
     *          @ref ral::lib::services::ReturnStatus::signal_received - In case a signal was received during the operation.
     */
    ReturnStatus get_next_chunk();
    /**
     * @brief: Sets chunk commit options. The options affect only the next commit.
     *
     * @param [in] options: A set of commmit options to enable.
     */
    void set_commit_options(const std::unordered_set<rmx_output_commit_option>& options);
    /**
     * @brief: Sets chunk destination address.
     *
     * @param [in] addr: Chunk destination address.
     *
     * @warning: It shall be called for every chunk of a stream,
     *           which upon creation wasn't set into a connected mode, i.e. for which
     *           using a fixed destination address was not configured.
     * @warning: This API shall be called for a chunk after @ref get_next_chunk but
     *           prior to @ref apply_packets_layout.
     */
    void set_dest_address(const sockaddr& addr) { rmx_output_gen_set_chunk_remote_addr(&m_chunk, &addr); }
    /**
     * @brief: Appplies memory layout (address, length) and destination address of all
     *         the packets in the chunk.
     *
     * @warinig: This function must be called before each @ref commit_chunk.
     *
     * @return: Status code as defined by @ref ReturnStatus.
     */
    ReturnStatus apply_packets_layout();
    /**
     * @brief: Sends a chunk of the associated generic output stream to the wire.
     *
     * @param [in] time: Time to schedule chunk sending, the format depends on the options
     * set in @ref set_commit_options.
     *
     * @warinig: If multiple chunks were acquired with @ref get_next_chunk for the
     * stream after the previous call to @ref commit_chunk, the oldest acquired chunk will
     * be sent, not the one whose method was called.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ral::lib::services::ReturnStatus::hw_send_queue_full - In case the HW send queue is full.
     *          @ref ral::lib::services::ReturnStatus::signal_received - In case a signal was received during the operation.
     */
    ReturnStatus commit_chunk(uint64_t time);
    /**
     * @brief: Marks the chunk for completion tracking.
     *
     * @param [out] token: Arbitrary user token associated with the chunk.
     *
     * @warinig: To mark a chunk, call this function after @ref get_next_chunk
     * but before committing it with @ref commit_chunk.
     *
     * @return Status code as defined by @ref rmx_status.
     */
    ReturnStatus mark_for_tracking(uint64_t token);
    /**
     * @brief: Polls for a new transmit completion of a chunk marked for completion tracking.
     *
     * @warning: When only some of the committed chunks were marked for tracking, this function may return
     *          @ref ral::lib::services::ReturnStatus::no_completion even if the marked chunks are already transmitted,
     *          but their status is not yet fetched from HW due to many preceeding non-marked completions.
     *          In this use case the function must be called multiple times until ral::lib::services::ReturnStatus::success
     *          is returned. Then the marked completion info can be read with @ref get_last_completion_info.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::no_completion - In case when no new completions are available.
     */
    ReturnStatus poll_for_completion();
    /**
     * @brief: Retrieves information of the last Tx completion obtained with @ref poll_for_completion.
     *
     * @param [out] timestamp: Time of completing the chunk transmit set by HW.
     * @param [out] token: Token associated with the chunk when marking for trackinw with @ref mark_for_tracking.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of incorrect or obsolete stream.
     */
    ReturnStatus get_last_completion_info(uint64_t& timestamp, uint64_t& token);
};

} // namespace core
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_CORE_CHUNK_GENERIC_CHUNK_H_
#endif /* RMAX_APPS_LIB_LIB_CORE_CHUNK_GENERIC_CHUNK_H_ */
