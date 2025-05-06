/*
 * Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef RMAX_APPS_LIB_IO_NODE_SENDERS_RTP_VIDEO_SEND_STREAM_H_
#define RMAX_APPS_LIB_IO_NODE_SENDERS_RTP_VIDEO_SEND_STREAM_H_

#include <cstddef>
#include <vector>
#include <unordered_map>
#include <memory>
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
constexpr size_t NS_IN_SEC = std::chrono::nanoseconds{ std::chrono::seconds{ 1 } }.count();
constexpr uint8_t LEAP_SECONDS = 37;
constexpr const char* VIDEO_2110_20_1080p50 = "1080p50";
constexpr const char* VIDEO_2110_20_1080p60 = "1080p60";
constexpr const char* VIDEO_2110_20_2160p50 = "2160p50";
constexpr const char* VIDEO_2110_20_2160p60 = "2160p60";
const std::unordered_set<const char*> SUPPORTED_STREAMS = {
    VIDEO_2110_20_1080p50,
    VIDEO_2110_20_1080p60,
    VIDEO_2110_20_2160p50,
    VIDEO_2110_20_2160p60
};

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
 * @brief: Compose @ref media_settings_t for the given media stream.
 *
 * This helper function calculates media settings for the selected video stream format.
 */
void compose_media_settings(AppSettings& s);
/**
 * @brief: Calculate timing parameters for the given media stream.
 *
 * This helper function calculates media settings for the selected video stream format.
 */
void calculate_tro_trs(media_settings_t& media_settings, double& tro, double& trs);
/**
 * @brief: RTP Video send stream interface.
 *
 * This class implements and extends @ref ral::lib::core::MediaSendStream operations.
 */
class RTPVideoSendStream : public MediaSendStream
{
private:
    SendingStatistics m_send_stats;
public:
    /**
     * @brief: RTPVideoSendStream constructor without assigning memory blocks.
     *
     * @param [in] settings: Stream parameters.
     */
    RTPVideoSendStream(const MediaStreamSettings& settings);
    /**
     * @brief: RTPVideoSendStream constructor with assigning memory blocks.
     *
     * @param [in] settings: Stream parameters.
     * @param [in] mem_blocks: Parameters of blocks allocated for output packets.
     */
    RTPVideoSendStream(const MediaStreamSettings& settings, MediaStreamMemBlockset& mem_blocks);
    virtual ~RTPVideoSendStream() = default;
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

#endif /* RMAX_APPS_LIB_IO_NODE_SENDERS_RTP_VIDEO_SEND_STREAM_H_ */
