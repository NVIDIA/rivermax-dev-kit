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

#ifndef RDK_APPS_RMAX_MEDIA_PROBE_STREAM_MONITOR_H_
#define RDK_APPS_RMAX_MEDIA_PROBE_STREAM_MONITOR_H_

#include <cstdint>
#include <cstddef>
#include <iostream>
#include <mutex>

#include "rdk/services/error_handling/return_status.h"
#include "rdk/apps/rmax_media_probe/media_monitor.h"

namespace rivermax
{
namespace dev_kit
{
namespace apps
{
namespace rmax_media_probe
{

struct MediaProbeSettings;

/**
 * @brief: Stream monitor for RTP packet processing and frame detection.
 *
 * This class monitors RTP streams, processes incoming packets, detects frame boundaries,
 * and tracks various statistics including packet loss, frame rates, and media delays.
 * It provides callbacks for new frame events and maintains shared statistics.
 */
class StreamMonitor
{
public:
    /**
     * @brief: StreamMonitor constructor.
     *
     * @param [in] app_settings: Application settings.
     * @param [in] component_index: Index of the media component this monitor handles.
     * @param [in] on_new_frame_callback: Callback function to invoke when a new frame is detected.
     */
    StreamMonitor(const MediaProbeSettings &app_settings, size_t stream_index, size_t component_index, OnNewFrameCallback on_new_frame_callback) :
        m_app_settings(app_settings),
        m_stream_index(stream_index),
        m_component_index(component_index),
        m_on_new_frame_callback(on_new_frame_callback) {}
    /**
     * @brief: StreamMonitor destructor.
     */
    virtual ~StreamMonitor() = default;
    /**
     * @brief: Process a received chunk of RTP packets.
     *
     * @param [in] chunk: The received chunk containing RTP packets.
     * @param [in] stream: The receive stream interface.
     * @param [out] consumed_packets: Number of packets consumed from the chunk.
     * @return: Return status indicating success or failure.
     */
    ReturnStatus consume_chunk(const ReceiveChunk &chunk, const IReceiveStream &stream, size_t &consumed_packets);
    /**
     * @brief: Extract RTP sequence number from packet header.
     *
     * @param [in] header: Pointer to the RTP packet header.
     * @param [in] length: Length of the header data.
     * @param [out] sequence_number: Extracted sequence number.
     * @return: True if sequence number was successfully extracted, false otherwise.
     */
    bool get_sequence_number(const byte_t *header, size_t length, uint32_t &sequence_number) const;
    /**
     * @brief: Extract RTP timestamp from packet header.
     *
     * @param [in] header: Pointer to the RTP packet header.
     * @return: RTP timestamp value.
     */
    uint32_t get_rtp_timestamp(const byte_t *header) const;
    /**
     * @brief: Print current statistics and reset counters.
     *
     * @param [out] os: Output stream to write statistics to.
     */
    void print_and_reset_stats(std::ostream& os);
protected:
    /**
     * @brief: Measure media delay between receive timestamp and RTP timestamp.
     *
     * @param [in] receive_ts: Timestamp when packet was received.
     * @param [in] rtp_ts: RTP timestamp from packet header.
     */
    void measure_media_delay(uint64_t receive_ts, uint32_t rtp_ts);
    /**
     * @brief: Process a newly detected frame.
     *
     * @param [in] receive_timestamp: Timestamp when frame was received.
     * @param [in] rtp_timestamp: RTP timestamp of the frame.
     * @param [in] rtp_seq_num: RTP sequence number of the frame.
     * @param [in] stream: Reference to the receive stream.
     */
    void process_new_frame(uint64_t receive_timestamp, uint32_t rtp_timestamp, uint32_t rtp_seq_num, const IReceiveStream& stream);
    /**
     * @brief: Update shared statistics with current values.
     */
    void update_shared_stats();
    const MediaProbeSettings& m_app_settings;
    size_t m_stream_index;
    size_t m_component_index;
    bool m_is_first_packet = true;
    uint32_t m_prev_frame_rtp_timestamp = 0;
    uint64_t m_prev_frame_receive_timestamp = 0;
    uint32_t m_prev_frame_seq_num = 0;
    int32_t m_prev_rtp_seq_num = 0;
    uint64_t m_received_packets = 0;
    uint64_t m_missing_packets = 0;
    uint64_t m_bad_rtp_headers = 0;
    uint64_t m_received_frames = 0;
    uint64_t m_packets_per_frame = 0;
    bool m_is_first_frame = true;
    float m_fps = 0.0f;
    float m_media_delay_usec = 0.0f;
    float m_media_delay_min_usec = 0.0f;
    float m_media_delay_max_usec = 0.0f;
    OnNewFrameCallback m_on_new_frame_callback;
    /**
     * @brief: Shared statistics structure for thread-safe access.
     */
    struct SharedStats {
        uint64_t total_received_packets = 0;
        uint64_t received_packets_diff = 0;
        uint64_t total_missing_packets = 0;
        uint64_t missing_packets_diff = 0;
        uint64_t total_bad_rtp_headers = 0;
        uint64_t bad_rtp_headers_diff = 0;
        uint64_t total_received_frames = 0;
        uint64_t received_frames_diff = 0;
        uint64_t packets_in_last_frame = 0;
        uint64_t prev_frame_receive_ts = 0;
        uint32_t prev_frame_rtp_timestamp = 0;
        uint32_t prev_frame_seq_num = 0;
        float fps = 0.0f;
        float media_delay_usec = 0.0f;
        float media_delay_min_usec = 0.0f;
        float media_delay_max_usec = 0.0f;
    };

    SharedStats m_shared_stats;
    std::mutex m_shared_stats_mutex;
};

/**
 * @brief: Wrapper class that adapts StreamMonitor to IReceiveDataConsumer interface.
 *
 * This wrapper allows StreamMonitor to be used as a data consumer in the
 * Rivermax receive framework by implementing the IReceiveDataConsumer interface.
 */
class StreamMonitorWrapper : public IReceiveDataConsumer
{
public:
    /**
     * @brief: StreamMonitorWrapper constructor.
     *
     * @param [in] stream_monitor: Reference to the StreamMonitor instance to wrap.
     */
    StreamMonitorWrapper(StreamMonitor& stream_monitor) :
        m_stream_monitor(stream_monitor) {}
    /**
     * @brief: StreamMonitorWrapper destructor.
     */
    virtual ~StreamMonitorWrapper() = default;
    /**
     * @brief: Consume a chunk of received data by delegating to the wrapped StreamMonitor.
     *
     * @param [in] chunk: The received chunk containing RTP packets.
     * @param [in] stream: The receive stream interface.
     * @param [out] consumed_packets: Number of packets consumed from the chunk.
     * @return: Return status indicating success or failure.
     */
    ReturnStatus consume_chunk(const ReceiveChunk &chunk,
                               const IReceiveStream &stream,
                               size_t &consumed_packets) override
    {
        return m_stream_monitor.consume_chunk(chunk, stream, consumed_packets);
    }
protected:
    StreamMonitor& m_stream_monitor;
};

} // namespace rmax_media_probe
} // namespace apps
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_APPS_RMAX_MEDIA_PROBE_STREAM_MONITOR_H_ */
