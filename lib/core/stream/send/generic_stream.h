/*
 * Copyright © 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
    pp_rate_t m_pp_rate;
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
        pp_rate_t pp_rate, size_t num_of_requested_chunks, size_t num_of_packets_in_chunk,
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
 * This class implements @ref ral::lib::core::ISendStream operations.
 * It uses Rivermax TX generic API.
 */
class GenericSendStream : public ISendStream
{
protected:
    GenericStreamSettings m_stream_settings;
    std::vector<std::shared_ptr<GenericChunk>> m_chunks;
    size_t m_next_chunk_to_send_index;
    rmx_output_gen_stream_params m_stream_params;
public:
    /**
     * @brief: GenericSendStream constructor.
     *
     * @param [in] settings: Stream parameters.
     */
    GenericSendStream(const GenericStreamSettings& settings);
    std::ostream& print(std::ostream& out) const override;
    virtual ReturnStatus create_stream() override;
    virtual ReturnStatus destroy_stream() override;
    /**
     * @brief: Returns a generic chunk.
     *
     * @note: The user can override this function in case
     *        different logic and data structures for chunk management needed.
     *
     * @param [in] index: Index of the chunk.
     *
     * @return: Pointer to the generic chunk.
     */
    virtual std::shared_ptr<GenericChunk> get_chunk(size_t index) { return m_chunks[index]; };
    /**
     * @brief: Returns the next free generic chunk.
     *
     * @note: The user can override this function in case
     *        different logic and data structures for chunk management needed.
     *
     * @param [in] chunk: Reference to pointer to the returned chunk.
     *
     * @return: Status of the operation.
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ral::lib::services::ReturnStatus::no_free_chunks - In case of insufficient available chunks.
     *          @ref ral::lib::services::ReturnStatus::signal_received - In case a signal was received during the operation.
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
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ral::lib::services::ReturnStatus::no_free_chunks - In case of insufficient available chunks.
     *          @ref ral::lib::services::ReturnStatus::signal_received - In case a signal was received during the operation.
     */
    virtual ReturnStatus blocking_get_next_chunk(std::shared_ptr<GenericChunk>& chunk, size_t retries = BLOCKING_CHUNK_RETRIES);
    /**
     * @brief: Initializes chunks layout using the specified memory region.
     *
     * @param [in] mem_region - Memory region for packets data.
     */
    virtual void initialize_chunks(const rmx_mem_region& mem_region);
    /**
     * @brief: Returns memory length needed for the stream.
     *
     * @return: Size in bytes of the memory length.
     */
    virtual size_t get_memory_length() const;
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
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ral::lib::services::ReturnStatus::hw_send_queue_full - In case the HW send queue is full.
     *          @ref ral::lib::services::ReturnStatus::signal_received - In case a signal was received during the operation.
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
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ral::lib::services::ReturnStatus::hw_send_queue_full - In case the HW send queue is full.
     *          @ref ral::lib::services::ReturnStatus::signal_received - In case a signal was received during the operation.
     */
    ReturnStatus blocking_commit_chunk(std::shared_ptr<GenericChunk> chunk, uint64_t time, size_t retries = BLOCKING_COMMIT_RETRIES);
};

} // namespace core
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_CORE_STREAM_SEND_GENERIC_STREAM_H_
#endif /* RMAX_APPS_LIB_LIB_CORE_STREAM_SEND_GENERIC_STREAM_H_ */
