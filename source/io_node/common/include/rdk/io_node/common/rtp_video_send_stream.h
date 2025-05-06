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

#ifndef RDK_IO_NODE_COMMON_RTP_VIDEO_SEND_STREAM_H_
#define RDK_IO_NODE_COMMON_RTP_VIDEO_SEND_STREAM_H_

#include <cstddef>
#include <iostream>
#include <ostream>
#include <chrono>

#include "rdk/services/utils/defs.h"
#include "rdk/core/stream/send/media_stream.h"

using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
{
namespace io_node
{
/**
 * @note: This module will be deprecated in the future.
 */

constexpr size_t SLEEP_THRESHOLD_NS = std::chrono::nanoseconds{ std::chrono::milliseconds{ 2 } }.count();

/**
 * @brief: Sending statistics struct.
 *
 * This struct will hold run time statistics of a stream.
 */
struct SendingStatistics
{
    uint32_t packet_counter;
    uint32_t rtp_sequence;
    uint32_t rtp_timestamp;
    uint8_t rtp_interlace_field_indicator;
    uint16_t line_number;
    uint16_t srd_offset;
};
/**
 * @brief: RTP Video send stream interface.
 *
 * This class implements and extends @ref MediaSendStream operations.
 */
class RtpVideoSendStream : public MediaSendStream
{
private:
    SendingStatistics m_send_stats;
public:
    /**
     * @brief: RtpVideoSendStream constructor without assigning memory blocks.
     *
     * @param [in] settings: Stream parameters.
     */
    RtpVideoSendStream(const MediaStreamSettings& settings);
    /**
     * @brief: RtpVideoSendStream constructor with assigning memory blocks.
     *
     * @param [in] settings: Stream parameters.
     * @param [in] mem_blocks: Parameters of blocks allocated for output packets.
     */
    RtpVideoSendStream(const MediaStreamSettings& settings, MediaStreamMemBlockset& mem_blocks);
    virtual ~RtpVideoSendStream() = default;
    std::ostream& print(std::ostream& out) const override;
    /**
     * @brief: Prepares media chunk to send.
     *
     * This method will prepare RTP header for the given chunk.
     * The media payload will be random payload.
     *
     * @param [in] chunk: Media chunk.
     *
     * @note: TODO: Remove this and add the appropriate buffer writer interface in the library for this.
     */
    void prepare_chunk_to_send(MediaChunk& chunk);
    /**
     * @brief: Calculates stream inter-packet interval Trs.
     *
     * This method calculates the time between adjacent packets according SMPTE 2110-21 spec.
     *
     * @return: Inter-packet interval.
     */
    double calculate_trs();
    /**
     * @brief: Calculates stream send time.
     *
     * This method will calculate the time to send the stream based on SPMTE 2110, from the given time plus 1 second.
     * It will also update with the appropriate rtp_timestamp in the sending statistics of the stream.
     *
     * @param [in] time_now_ns: Send time in nanosecond starting point.
     *
     * @note: TODO: Remove this and add send time calculation logic components in the library.
     *
     * @return: Time to send the stream in nanoseconds.
     */
    double calculate_send_time_ns(uint64_t time_now_ns);
private:
    /**
     * @brief: Builds 2110-20 RTP header.
     *
     * @param [in] buffer: Pointer to the buffer of the packet to fill.
     *
     * @note: TODO: Remove this and add stream buffer writer for RTP components in the library.
     */
    inline void build_2110_20_rtp_header(byte_t* buffer);
};

} // io_node
} // dev_kit
} // rivermax

#endif /* RDK_IO_NODE_COMMON_RTP_VIDEO_SEND_STREAM_H_ */
