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

#ifndef RMAX_APPS_LIB_LIB_CORE_STREAM_SEND_MEDIA_STREAM_H_

#include <cstddef>
#include <string>
#include <memory>
#include <ostream>
#include <vector>

#include <rivermax_api.h>

#include "core/stream/send/send_stream_interface.h"
#include "core/flow/flow.h"
#include "core/chunk/media_chunk.h"
#include "services/utils/defs.h"
#include "services/error_handling/return_status.h"

namespace ral
{
namespace lib
{
namespace core
{
/**
 * @brief: Media API send stream interface
 *
 * This class implements @ref ral::lib::core::ISendStream operations.
 * It uses Rivermax TX media API.
 */
class MediaSendStream : public ISendStream
{
protected:
    media_settings_t m_media_settings;
private:
    rmax_out_stream_params m_rmax_parameters;
public:
    /**
     * @brief: MediaSendStream constructor.
     *
     * @param [in] local_address: Network address of the NIC.
     * @param [in] media_settings: Media related settings.
     * @param [in] buffer_attributes: Pointer to buffer attributes, for more info
     *                                see @ref rmax_buffer_attr in rivermax_api.h.
     * @param [in] qos_attributes: Pointer to quality of service attributes, for more info
     *                             see @ref rmax_qos_attr in rivermax_api.h.
     */
    MediaSendStream(
        const TwoTupleFlow& local_address, const media_settings_t& media_settings,
        rmax_buffer_attr* buffer_attributes, rmax_qos_attr* qos_attributes);
    std::ostream& print(std::ostream& out) const override;
    virtual ReturnStatus create_stream() override;
    virtual ReturnStatus destroy_stream() override;
    /**
     * @brief: Returns data stride size of the stream buffer attributes.
     *
     * @return: Data stride size.
     */
    virtual uint16_t get_data_stride_size() const { return m_rmax_parameters.buffer_attr->data_stride_size; }
    /**
     * @brief: Returns next media chunk.
     *
     * @param [out] chunk: Reference to the returned chunk.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ral::lib::services::ReturnStatus::no_free_chunks - In case of insufficient available chunks.
     */
    virtual ReturnStatus get_next_chunk(MediaChunk& chunk);
    /**
     * @brief: Returns next media chunk - blocking operation.
     *
     * @param [out] chunk: Pointer to the returned chunk.
     * @param [in] retries: Number of retries, in case of insufficient available chunks.
     *                      Defaults to @ref ral::lib::core::BLOCKING_CHUNK_RETRIES.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ral::lib::services::ReturnStatus::no_free_chunks - In case of insufficient available chunks.
     */
    virtual ReturnStatus blocking_get_next_chunk(MediaChunk& chunk, size_t retries = BLOCKING_CHUNK_RETRIES);
    /**
     * @brief: Sends the chunk.
     *
     * @param [in] timestamp_ns: Time in nanoseconds, see @ref rmax_out_commit in rivermax_api.h.
     * @param [in] flags: Commit flags, see @ref rmax_out_commit_flags in rivermax_api.h.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ral::lib::services::ReturnStatus::hw_send_queue_full - In case the HW send queue is full.
     */
    virtual ReturnStatus commit_chunk(uint64_t timestamp_ns, rmax_commit_flags_t flags) const;
    /**
     * @brief: Sends the chunk - blocking operation.
     *
     * @param [in] timestamp_ns: Time in nanoseconds, see @ref rmax_out_commit in rivermax_api.h.
     * @param [in] flags: Commit flags, see @ref rmax_out_commit_flags in rivermax_api.h.
     * @param [in] retries: Number of retries, in case the HW send queue is full.
     *                      Defaults to @ref ral::lib::core::BLOCKING_COMMIT_RETRIES.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     *          @ref ral::lib::services::ReturnStatus::hw_send_queue_full - In case the HW send queue is full.
     */
    virtual ReturnStatus blocking_commit_chunk(
        uint64_t timestamp_ns, rmax_commit_flags_t flags, size_t retries = BLOCKING_COMMIT_RETRIES) const;
private:
    /**
     * @brief: Initializes underlay Rivermax stream parameters.
     *
     * @param [in] buffer_attributes: Pointer to buffer attributes.
     * @param [in] qos_attributes: Pointer to quality of service attributes.
     */
    void initialize_rmax_stream_parameters(rmax_buffer_attr* buffer_attributes, rmax_qos_attr* qos_attributes);
};

} // namespace core
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_CORE_STREAM_SEND_MEDIA_STREAM_H_
#endif /* RMAX_APPS_LIB_LIB_CORE_STREAM_SEND_MEDIA_STREAM_H_ */
