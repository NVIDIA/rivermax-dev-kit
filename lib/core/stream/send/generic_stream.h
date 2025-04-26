/*
 * Copyright © 2017-2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef RMAX_APPS_LIB_LIB_CORE_STREAM_SEND_GENERIC_STREAM_H_

#include <cstddef>
#include <string>
#include <ostream>
#include <vector>

#include <rivermax_api.h>

#include "core/stream/send/send_stream_interface.h"
#include "core/flow/flow.h"
#include "core/chunk/generic_chunk.h"
#include "services/utils/defs.h"
#include "services/error_handling/return_status.h"
#include "services/memory_management/memory_management.h"

namespace ral
{
namespace lib
{
namespace core
{

constexpr int PACKET_IOVEC_SIZE = 1;
constexpr size_t OUT_STREAM_SIZE_DEFAULT = 32768;

/**
 * @brief: Generic API send stream interface
 *
 * This class implements @ref ral::lib::core::ISendStream operations.
 * It uses Rivermax TX generic API.
 */
class GenericSendStream : public ISendStream
{
private:
    const ral::lib::core::TwoTupleFlow m_destination_flow;

    rmax_out_gen_stream_params m_rmax_parameters;
    std::vector<GenericChunk*> m_chunks;
    size_t m_next_chunk_to_send_index;
    size_t m_num_of_requested_chunks;
    size_t m_num_of_packets_in_chunk;
    uint16_t m_packet_payload_size;
    uint16_t m_packet_app_header_size;
    gs_mem_block_t m_mem_block;
    pp_rate_t m_rate;
public:
    /**
     * @brief: GenericSendStream constructor.
     *
     * @param [in] network_address: Network address of the stream.
     * @param [in] local_address: Network address of the NIC.
     * @param [in] pp_rate: Packet pacing rate for the stream.
     *                      If packet pacing is not needed, the struct should be initialized to 0.
     * @param [in] num_of_requested_chunks: Number of chunks to be used in the stream.
     * @param [in] packet_typical_payload_size: Packet typical payload size in bytes.
     * @param [in] packet_typical_app_header_size: Packet typical application header size in bytes.
     */
    GenericSendStream(
        const FourTupleFlow& network_address,
        pp_rate_t pp_rate, size_t num_of_requested_chunks, size_t num_of_packets_in_chunk,
        uint16_t packet_typical_payload_size, uint16_t packet_typical_app_header_size);
    std::ostream& print(std::ostream& out) const override;
    virtual ReturnStatus create_stream() override;
    virtual ReturnStatus destroy_stream() override;
    /**
     * @brief: Returns generic chunk.
     *
     * @note: The user can override this function in case
     *        different logic and data structures for chunk management needed.
     *
     * @param [in] index: Index of the chunk.
     *
     * @return: Pointer to the generic chunk.
     */
    virtual GenericChunk* get_chunk(size_t index) { return m_chunks[index]; };
    /**
     * @brief: Returns next generic chunk.
     *
     * @note: The user can override this function in case
     *        different logic and data structures for chunk management needed.
     *
     * @param [out] chunk: Pointer to the returned chunk.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus get_next_chunk(GenericChunk* chunk);
    /**
     * @brief: Initializes chunks layout.
     *
     * @note: The user can override this function in case
     *        different logic and data structures for chunk management needed.
     *
     * @param [in] pointer: Pointer to the allocated memory for the chunk, this memory should be registered first.
     * @param [in] mkey: Memory key from @ref rmax_deregister_memory in Rivermax API.
     *
     * @return: Size of the memory initialized.
     */
    virtual size_t initialize_chunks(void* pointer, rmax_mkey_id mkey);
    /**
     * @brief: Returns memory length needed for the stream.
     *
     * @return: Size in bytes of the memory length.
     */
    virtual size_t get_memory_length() const;
    /**
     * @brief: Sends the chunk - blocking operation.
     *
     * @param [in] chunk: Reference of the chunk to commit.
     * @param [in] timestamp_ns: Time in nanoseconds when to start transmission of the chunk.
     * @param [in] flags: Commit flags, see @ref rmax_out_commit_flags in rivermax_api.h.
     * @param [in] dest_flow: Pointer to network detestation flow, defaults to nullptr.
     *     @note: In case of nullptr, the destination flow will be the same as the flow the stream was initialized with.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus blocking_commit_chunk(
        GenericChunk& chunk, uint64_t timestamp_ns, rmax_commit_flags_t flags, TwoTupleFlow* dest_flow = nullptr) const;
private:
    /**
     * @brief: Initializes underlay Rivermax stream parameters.
     */
    void initialize_rmax_stream_parameters();
    /**
     * @brief: Commit chunk helper - blocking operation.
     *
     * @param [in] chunk: Pointer to Rivermax generic chunk to commit.
     * @param [in] timestamp_ns: Time in nanoseconds when to start transmission of the chunk.
     * @param [in] flags: Commit flags, see @ref rmax_out_commit_flags in rivermax_api.h.
     * @param [in] flow: Network destination address for the commit, defaults to nullptr.
     * @param [in] retries: Number of retries for the blocking operation, defaults to BLOCKING_COMMIT_RETRIES.
     *
     * @note: The destination flow will be the destination flow the stream was initialized with.
     *
     * @return: Status of the operation.
     */
    inline ReturnStatus blocking_commit_chunk_helper(
        rmax_chunk* chunk, uint64_t timestamp_ns, rmax_commit_flags_t flags,
        sockaddr* flow = nullptr, size_t retries = BLOCKING_COMMIT_RETRIES) const;
};

} // namespace core
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_CORE_STREAM_SEND_GENERIC_STREAM_H_
#endif /* RMAX_APPS_LIB_LIB_CORE_STREAM_SEND_GENERIC_STREAM_H_ */
