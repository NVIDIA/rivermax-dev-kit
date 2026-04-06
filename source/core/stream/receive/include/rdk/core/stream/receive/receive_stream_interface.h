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

#ifndef RDK_CORE_STREAM_RECEIVE_RECEIVE_STREAM_INTERFACE_H_
#define RDK_CORE_STREAM_RECEIVE_RECEIVE_STREAM_INTERFACE_H_

#include <cstddef>
#include <ostream>
#include <chrono>

#include <rivermax_api.h>

#include "rdk/core/memory_layout/header_payload_memory_layout.h"
#include "rdk/core/stream/stream_interface.h"
#include "rdk/core/chunk/receive_chunk.h"

namespace rdk
{
namespace core
{
/**
 * @brief: Receive stream interface.
 */
class IReceiveStream : public IStream, public IHeaderPayloadMemoryLayoutComponent
{
public:
    virtual ~IReceiveStream() = default;
    /**
     * @brief: Detaches flows from all redundant streams.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    virtual ReturnStatus detach_flows() = 0;
    /**
     * @brief: Applies the runtime parameters of the stream.
     *
     * This method is responsible to update the stream runtime parameters with the values
     * provided in @ref ReceiveStreamSettings.
     *
     * @return: Status of the operation.
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    virtual ReturnStatus apply_runtime_parameters() = 0;
    /**
     * @brief: Configures completion moderation parameters.
     *
     * This method configures the rule for how many packets to receive
     * or how long to wait before returning the next requested chunk.
     *
     * @param [in] min_count: A minimal number of packets to return.
     * @param [in] max_count: A maximal number of packets to return.
     * @param [in] timeout_us: A timeout in microseconds to wait for @p min_count of packets.
     *
     * @note: Setting @p timeout_us to 0 results in Rivermax busy-looping until
     *        @p max_count packets are received.
     * @note: To avoid Rivermax waiting for packets in a busy-loop, set both
     *        @p min_count and @p max_count to 0.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    virtual ReturnStatus set_completion_moderation(size_t min_count, size_t max_count, int timeout_us) = 0;
    /**
     * @brief: Returns next chunk from the stream.
     *
     * @param [out] chunk: A chunk received from the stream.
     *
     * @return: Status code as defined by @ref ReturnStatus.
     *          @ref ReturnStatus::success - In case of success.
     *          @ref ReturnStatus::signal_received - If operation was interrupted by an OS signal.
     *          @ref ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    virtual ReturnStatus get_next_chunk(ReceiveChunk& chunk) = 0;
    /**
     * @brief: Returns header stride size.
     *
     * @return: Stride size in bytes.
     */
    virtual size_t get_header_stride_size() const = 0;
    /**
     * @brief: Returns payload stride size.
     *
     * @return: Stride size in bytes.
     */
    virtual size_t get_payload_stride_size() const = 0;
    /**
     * @brief: Returns Header-Data-Split mode status.
     *
     * @return: Header-Data-Split mode status.
     */
    virtual bool is_header_data_split_on() const = 0;
    /**
     * @brief: Returns the stream name.
     *
     * @return: Stream name.
     */
    virtual std::string get_stream_name() const = 0;
    /**
     * @brief: Prints stream statistics.
     *
     * @param [out] out: Output stream to print statistics to.
     * @param [in] interval_duration: Statistics interval duration.
     */
    virtual void print_statistics(std::ostream& out, const std::chrono::high_resolution_clock::duration& interval_duration) const {};
    /**
     * @brief: Resets statistics.
     */
    virtual void reset_statistics() {};
    /**
     * @brief: Resets totals statistics.
     */
    virtual void reset_statistics_totals() {};
protected:
    /**
     * @brief: IReceiveStream class constructor.
     */
    IReceiveStream() : IStream() {}
};

} // namespace core
} // namespace rdk

#endif /* RDK_CORE_STREAM_RECEIVE_RECEIVE_STREAM_INTERFACE_H_ */
