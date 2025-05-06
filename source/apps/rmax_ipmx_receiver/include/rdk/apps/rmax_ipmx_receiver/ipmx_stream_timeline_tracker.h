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

#ifndef RDK_APPS_RMAX_IPMX_RECEIVER_IPMX_STREAM_TIMELINE_TRACKER_H_
#define RDK_APPS_RMAX_IPMX_RECEIVER_IPMX_STREAM_TIMELINE_TRACKER_H_

#include <atomic>
#include <chrono>

#include "rdk/services/utils/defs.h"
#include "rdk/services/utils/static_rcu.h"
#include "rdk/services/error_handling/error_handling.h"
#include "rdk/services/media/ipmx.h"
#include "rdk/core/chunk/receive_chunk.h"

using namespace rivermax::dev_kit::core;
using namespace rivermax::dev_kit::services;

namespace rivermax
{
namespace dev_kit
{
namespace apps
{
namespace rmax_ipmx_receiver
{
/**
 * @brief: Tracks IPMX stream timing between sender and receiver using RTP/RTCP protocol data.
 *
 * This class tracks the timing of the composite IPMX stream by processing the events of its
 * component streams: receiving Sender Reports in the RTCP stream and detecting new media frames
 * in the RTP stream. It checks the matching of the sequence of Sender Reports and media Frames,
 * measures time properties of the Sender clock against the Receiver clock.
 * It provides the current stream media parameters obtained from the sender reports,
 * timing measurement statistics and can be extended to implement sender clock recovery.
 */
 class IPMXStreamTimelineTracker
{
protected:
    /**
     * @brief: Statistics about the received RTCP Sender Reports.
     */
    struct RTCPReportStats {
        size_t reports_received = 0;
        size_t reports_corrupted = 0;
        uint64_t last_report_rx_ts = 0;
        bool is_last_report_valid = false;
    };
    /**
     * @brief: State of RTCP Sender Report processing.
     */
    struct IPMXSenderReportState {
        IPMXStreamInfo last_report;
        RTCPReportStats stats;
        /**
         * @brief: Parses a RTCP Sender Report paket, stores the results and updates the statistics.
         *
         * @param [in] ipmx_stream_id: IPMX Stream Id for for logging.
         * @param [in] data: Pointer to the RTCP packet data.
         * @param [in] info: Packet meta-information.
         */
        void update(uint32_t ipmx_stream_id, const byte_t* data, const ReceivePacketInfo& info);
    };
    /**
     * @brief: A timing record of one media frame.
     */
    struct FrameTiming {
        uint32_t tx_rtp_timestamp = 0;
        uint64_t tx_ntp_timestamp = 0;
        uint64_t rx_timestamp = 0;
        int64_t clock_diff = 0;
    };
    /**
     * @brief: Clock measurement statistics.
     */
    struct ClockStats {
        size_t frames_received = 0;
        size_t frames_missing_send_report = 0;
        size_t frames_bad_rtp_time = 0;
        int64_t clock_drift_since_start = 0;
        int64_t clock_drift_last_period = 0;
        double clock_ratio_ppm_since_start = 0;
        double clock_ratio_ppm_last_period = 0;
        int64_t sender_report_to_data_min = 0;
        int64_t sender_report_to_data_max = 0;
    };
    /**
     * @brief: State of the data clock tracking.
     */
    struct IPMXDataClockState {
        FrameTiming first_frame;
        FrameTiming last_frame;
        FrameTiming prev_stats_frame;
        ClockStats stats;
        /**
         * @brief: Processes a new RTP media frame event and recalculates clock parameters.
         *
         * @param [in] ipmx_stream_id: Stream Id for for logging.
         * @param [in] report_state: The most recent update of Sender Report.
         * @param [in] rtp_timestamp: RTP timestamp of the frame.
         * @param [in] receiver_timestamp: Receiver clock at the moment of frame reception.
         * @param [in] need_update_stats: Requests to refresh the clock statistics.
         */
         void update(uint32_t ipmx_stream_id,
            const IPMXSenderReportState& report_state,
            uint32_t rtp_timestamp, uint64_t receiver_timestamp,
            bool need_update_stats);
    };
    uint32_t m_ipmx_stream_id;
    IPMXSenderReportState m_local_report_state;
    IPMXDataClockState m_local_clock_state;
    static constexpr int RCU_NUM_VERSIONS = 4;
    StaticRCU<IPMXSenderReportState, RCU_NUM_VERSIONS> m_shared_report_state;
    StaticRCU<IPMXDataClockState, RCU_NUM_VERSIONS> m_shared_clock_state;
    std::atomic<bool> m_new_report_version_available;
    std::atomic<bool> m_new_clock_stats_available;
    uint32_t m_stats_update_interval_ms;
public:
    /**
     * @brief: IPMXStreamTimelineTracker constructor.
     *
     * @param [in] ipmx_stream_id: Application level stream ID.
     */
    IPMXStreamTimelineTracker(uint32_t ipmx_stream_id);
    virtual ~IPMXStreamTimelineTracker() = default;
    /**
     * @brief: Processes an RTCP packet with an RTCP Sender Report.
     *
     * @param [in] data: Pointer to the network packet data.
     * @param [in] info: Metainformation of the received packet.
     */
    void consume_rtcp_packet(const byte_t *data, const ReceivePacketInfo& info);
    /**
     * @brief: Processes an event of start of a new media frame.
     *
     * @param [in] rtp_timestamp: RTP timestamp encoded in the frame.
     * @param [in] receiver_timestamp: Timestamp of the first frame packet receiving.
     */
    void notify_data_frame(uint32_t rtp_timestamp, uint64_t receiver_timestamp);
    /**
     * @brief: Prints the latest update in stream parameters and statistics.
     */
    void print_updates();
    /**
     * @brief: Sets the interval for updating statistic reports.
     *
     * @param [in] interval_ms: Interval for statistics update.
     */
    void set_statistics_report_interval(uint32_t interval_ms);
protected:
    /**
     * @brief: Prints the latest media stream information obtained from Sender reports.
     *
     * @param [in] out: Output stream.
     */
     void print_stream_new_info(std::ostream& out);
    /**
     * @brief: Prints the latest update ou data clock measurements and staistics.
     *
     * @param [in] out: Output stream.
     */
     void print_clock_new_info(std::ostream& out);
};

} // namespace rmax_ipmx_receiver
} // namespace apps
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_APPS_RMAX_IPMX_RECEIVER_IPMX_STREAM_TIMELINE_TRACKER_H_ */
