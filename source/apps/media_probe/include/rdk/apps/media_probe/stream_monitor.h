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

#ifndef RDK_APPS_MEDIA_PROBE_STREAM_MONITOR_H_
#define RDK_APPS_MEDIA_PROBE_STREAM_MONITOR_H_

#include <cstdint>
#include <cstddef>
#include <iostream>
#include <mutex>

#include "rdk/core/stream/receive/receive_stream_interface.h"
#include "rdk/core/flow/receive_flow.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/protocol/media_packet_parser.h"
#include "rdk/core/data_handler/receive_data_consumer_interface.h"

using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
{
namespace apps
{
namespace media_probe
{

/**
 * @brief: Media component identifiers.
 */
enum class MediaComponentId : size_t
{
    Video = 0,
    Alpha = 1,
    Count = 2
};

/**
 * @brief: Event structure containing information about a newly detected frame.
 */
struct NewFrameEvent
{
    MediaComponentId component_id;
    const IReceiveStream& stream;
    uint64_t receive_ts;
    uint32_t rtp_ts;
    uint32_t rtp_seq_num;
    float media_latency_usec;
    /**
     * @brief: Constructor for NewFrameEvent.
     *
     * @param [in] component_id: Media component identifier.
     * @param [in] stream: Reference to the receive stream.
     * @param [in] receive_ts: Timestamp when frame was received.
     * @param [in] rtp_ts: RTP timestamp of the frame.
     * @param [in] rtp_seq_num: RTP sequence number of the frame.
     * @param [in] media_latency_usec: Media latency in microseconds.
     */
    NewFrameEvent(MediaComponentId component_id,
                  const IReceiveStream& stream,
                  uint64_t receive_ts,
                  uint32_t rtp_ts,
                  uint32_t rtp_seq_num,
                  float media_latency_usec) :
        component_id(component_id),
        stream(stream),
        receive_ts(receive_ts),
        rtp_ts(rtp_ts),
        rtp_seq_num(rtp_seq_num),
        media_latency_usec(media_latency_usec)
    {
    }
};

/**
 * @brief: Callback function type for new frame events.
 */
using OnNewFrameCallback = std::function<void (const NewFrameEvent& event)>;

const OnNewFrameCallback null_new_frame_callback;

struct MediaProbeSettings;

/**
 * @brief: Stream monitor for RTP packet processing and frame detection.
 *
 * This class monitors RTP streams, processes incoming packets, detects frame boundaries,
 * and tracks various statistics including packet loss, frame rates, and media latencies.
 * It provides callbacks for new frame events and maintains shared statistics.
 */
class StreamMonitor
{
protected:
    const MediaProbeSettings& m_app_settings;
    const ReceiveFlow m_flow;
    MediaPacketParser m_packet_parser;
    size_t m_stream_index;
    MediaComponentId m_component_id;
    bool m_is_first_packet = true;
    uint32_t m_prev_frame_rtp_timestamp = 0;
    uint64_t m_prev_frame_receive_timestamp = 0;
    uint32_t m_prev_frame_seq_num = 0;
    uint32_t m_prev_rtp_seq_num = 0;
    bool m_is_prev_mbit_set = false;
    uint64_t m_received_packets = 0;
    uint64_t m_missing_packets = 0;
    uint64_t m_bad_rtp_headers = 0;
    uint64_t m_received_frames = 0;
    uint64_t m_packets_per_frame = 0;
    bool m_is_first_frame = true;
    float m_fps = 0.0f;
    float m_media_latency_usec = 0.0f;
    float m_media_latency_min_usec = 0.0f;
    float m_media_latency_max_usec = 0.0f;
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
        float media_latency_usec = 0.0f;
        float media_latency_min_usec = 0.0f;
        float media_latency_max_usec = 0.0f;
    };

    SharedStats m_shared_stats;
    std::mutex m_shared_stats_mutex;
public:
    /**
     * @brief: StreamMonitor constructor.
     *
     * @param [in] app_settings: Application settings.
     * @param [in] component_id: Id of the media component this monitor handles.
     */
    StreamMonitor(const MediaProbeSettings &app_settings, const ReceiveFlow& flow, size_t stream_index, MediaComponentId component_id);
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
     * @brief: Print current statistics and reset counters.
     *
     * @param [out] out: Output stream to write statistics to.
     */
    void print_and_reset_stats(std::ostream& out);
    /**
     * @brief: Set a new frame callback.
     *
     * @param [in] on_new_frame_callback: Callback function to invoke when a new frame is detected.
     */
    void set_on_new_frame_callback(OnNewFrameCallback on_new_frame_callback) { m_on_new_frame_callback = std::move(on_new_frame_callback); };
    /**
     * @brief: Get the media component Id of the stream monitor.
     *
     * @return: Media component Id of the stream monitor.
     */
    MediaComponentId get_component_id() const { return m_component_id; };
protected:
    /**
     * @brief: Measure media latency as difference between receive timestamp and RTP timestamp.
     *
     * @param [in] receive_ts: Timestamp when packet was received.
     * @param [in] rtp_ts: RTP timestamp from packet header.
     */
    void measure_media_latency(uint64_t receive_ts, uint32_t rtp_ts);
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
};

/**
 * @brief: Adapter class that adapts StreamMonitor to IReceiveDataConsumer interface.
 *
 * This adapter allows StreamMonitor to be used as a data consumer in the
 * Rivermax Dev Kit receive framework by implementing the IReceiveDataConsumer interface.
 */
class StreamMonitorAdapter : public IReceiveDataConsumer
{
public:
    /**
     * @brief: StreamMonitorAdapter constructor.
     *
     * @param [in] stream_monitor: Reference to the StreamMonitor instance to wrap.
     */
    StreamMonitorAdapter(StreamMonitor& stream_monitor) :
        m_stream_monitor(stream_monitor) {}
    /**
     * @brief: StreamMonitorWrapper destructor.
     */
    virtual ~StreamMonitorAdapter() = default;
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

} // namespace media_probe
} // namespace apps
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_APPS_MEDIA_PROBE_STREAM_MONITOR_H_ */
