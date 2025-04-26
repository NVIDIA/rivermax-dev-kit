/*
 * Copyright © 2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef RMAX_APPS_LIB_LIB_CORE_STREAM_RECEIVE_RECEIVE_STREAM_H_

#include <string>
#include <vector>
#include <unordered_map>

#include <rivermax_api.h>

#include "core/chunk/receive_chunk.h"
#include "core/stream/single_stream_interface.h"
#include "services/utils/defs.h"
#include "services/error_handling/return_status.h"

namespace ral
{
namespace lib
{
namespace core
{

/**
 * @brief: Base RX stream class.
 *
 * This class implements @ref ral::lib::core::ISingleStream operations.
 * Wraps up Rivermax stream object and flow.
 */
class ReceiveStream: public ISingleStream
{
public:
    /**
     * @brief: Receive stream constructor.
     *
     * @param [in] rx_type: Stream steering type, see @ref rmax_in_stream_type in rivermax_api.h.
     * @param [in] local_addr: NIC address.
     * @param [in] settings: Common input stream settings.
     */
    ReceiveStream(rmax_in_stream_type rx_type, const TwoTupleFlow& local_addr,
            const receive_stream_settings_t& settings);
    /**
     * @brief: Destroy receive stream.
     */
    virtual ~ReceiveStream() = default;
    /**
     * @brief: Print detailed stream information.
     *
     * @param [out] out: Output stream to print information.
     *
     * @return: Output stream.
     */
    std::ostream& print(std::ostream& out) const override;
    /**
     * @brief: Creates the stream.
     *
     * This method is responsible to create Rivermax stream.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus create_stream() override;
    /**
     * @brief: Attach a flow to the stream.
     *
     * @param [in] flow: Flow to attach to.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus attach_flow(const FourTupleFlow& flow);
    /**
     * @brief: Detach a flow from the stream.
     *
     * @param [in] flow: Flow to detach from.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus detach_flow(const FourTupleFlow& flow);
    /**
     * @brief: Destroys the stream.
     *
     * This method is responsible to destroy Rivermax stream.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus destroy_stream() override;
    /**
     * @brief: Receive next chunk from input stream.
     *
     * @param [out] chunk: Pointer to the returned chunk structure.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::signal_received - If operation was interrupted by an OS signal.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus get_next_chunk(ReceiveChunk* chunk);
    /**
     * @brief: Return the memory size needed by Rivermax to create stream with
     * given parameters.
     *
     * @param [out] header_size: Header buffer size (if header-data split is
     *                          enabled).
     * @param [out] payload_size: Payload buffer size.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus query_buffer_size(size_t& header_size, size_t& payload_size);
    /**
     * @brief: Set buffers for header and payload data.
     *
     * This method is optional. By default Rivermax will allocate memory
     * buffer(s) internally.
     *
     * @param [in] header_ptr: Pointer to header data buffer. Should be NULL if
     *                         header-data split mode is not enabled.
     * @param [in] payload_ptr: Pointer to payload data buffer.
     */
    void set_buffers(void* header_ptr, void* payload_ptr);
    /**
     * @brief: Query header stride size.
     *
     * @return: Stride size in bytes.
     */
    uint16_t get_header_stride_size() const { return m_header_block.stride_size; }
    /**
     * @brief: Query payload stride size.
     *
     * @return: Stride size in bytes.
     */
    uint16_t get_payload_stride_size() const { return m_payload_block.stride_size; }

private:
    const rmax_in_stream_type m_rx_type;
    const receive_stream_settings_t m_settings;

    std::unordered_map<FourTupleFlow, rmax_in_flow_attr> m_flows;
    rmax_in_buffer_attr m_buffer_attr;
    rmax_in_memblock m_header_block;
    rmax_in_memblock m_payload_block;
    rmax_in_timestamp_format m_timestamp_format;
};

} // namespace core
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_CORE_STREAM_RECEIVE_RECEIVE_STREAM_H_
#endif /* RMAX_APPS_LIB_LIB_CORE_STREAM_RECEIVE_RECEIVE_STREAM_H_ */
