/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef RMAX_APPS_LIB_LIB_CORE_STREAM_RTP_VIDEO_SEND_STREAM_H_
#define RMAX_APPS_LIB_LIB_CORE_STREAM_RTP_VIDEO_SEND_STREAM_H_

#include <cstddef>
#include <iostream>
#include <ostream>
#include <chrono>

#include <rivermax_api.h>

#include "api/rmax_apps_lib_api.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

namespace ral
{
namespace io_node
{

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
 * This class implements and extends @ref ral::lib::core::MediaSendStream operations.
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
} // ral

#endif /* RMAX_APPS_LIB_LIB_CORE_STREAM_RTP_VIDEO_SEND_STREAM_H_ */
