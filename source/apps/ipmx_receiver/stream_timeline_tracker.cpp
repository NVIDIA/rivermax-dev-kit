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

#include <cstring>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <memory>
#include <chrono>
#include <sstream>
#include <string>

#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/error_handling/error_handling.h"
#include "rdk/services/utils/clock.h"
#include "rdk/services/utils/counter_math.h"
#include "rdk/apps/ipmx_receiver/stream_timeline_tracker.h"

using namespace rdk::apps;

IPMXStreamTimelineTracker::IPMXStreamTimelineTracker(uint32_t ipmx_stream_id) :
    m_ipmx_stream_id(ipmx_stream_id),
    m_new_report_version_available(false),
    m_new_clock_stats_available(false),
    m_stats_update_interval_ms(std::chrono::milliseconds{ std::chrono::seconds{ 1 } }.count())
{
}

void IPMXStreamTimelineTracker::print_stream_new_info(std::ostream& out)
{
    IPMXSenderReportState current_report_state;
    ReturnStatus status = m_report_queue.retrieve_latest(current_report_state);
    if (status != ReturnStatus::success) {
        out << "IPMX Stream " << m_ipmx_stream_id << ": no Sender Report available" << std::endl;
        return;
    }
    auto& media = current_report_state.last_report.media.video;
    float fps = media.rate_num;
    if (media.rate_denom) {
        fps /= media.rate_denom;
    }
    auto old_prec = out.precision(3);
    out << "\n-----------------------------------------------------\n"
        << "IPMX Stream " << m_ipmx_stream_id << ": " << media.range << " "
        << media.width << "x" << media.height << (media.is_interlaced ? 'i' : 'p')
        << " " << fps << "fps " << std::to_string(media.bit_depth) << "bpp "
        << media.sampling_format << "\n"
        << "refclk=" << current_report_state.last_report.ts_refclk << "\n"
        << "mediaclk=" << current_report_state.last_report.mediaclk << "\n"
        << "color=" << media.colorimetry << " " << "tcs=" << media.tcs << "\n"
        << "cname=" << current_report_state.last_report.cname << "\n" << std::endl;
    out.precision(old_prec);
}

void IPMXStreamTimelineTracker::IPMXSenderReportState::update(uint32_t ipmx_stream_id,
    const byte_t* data, const ReceivePacketInfo& info)
{
    stats.reports_received++;
    stats.last_report_rx_ts = info.get_packet_timestamp();
    size_t len = info.get_packet_sub_block_size(0);

    auto status = last_report.fill_from_rtcp_sr_packet(data, len);
    if (status == ReturnStatus::success) {
        stats.is_last_report_valid = true;
    } else {
        stats.is_last_report_valid = false;
        stats.reports_corrupted++;
    }
}

void IPMXStreamTimelineTracker::consume_rtcp_packet(const byte_t* data, const ReceivePacketInfo& info)
{
    bool is_prev_report_valid = m_local_report_state.stats.is_last_report_valid;
    auto prev_info_block_version =  m_local_report_state.last_report.info_block_version;

    m_local_report_state.update(m_ipmx_stream_id, data, info);
    m_report_queue.add_version(m_local_report_state.last_report.rtp_ts, m_local_report_state);

    bool is_first_report = m_local_report_state.stats.is_last_report_valid &&
                           !is_prev_report_valid;
    bool is_media_info_changed = m_local_report_state.stats.is_last_report_valid &&
                                 (m_local_report_state.last_report.info_block_version !=
                                  prev_info_block_version);

    if (is_first_report || is_media_info_changed) {
        m_new_report_version_available.store(true);
    }
}

void IPMXStreamTimelineTracker::IPMXDataClockState::update(uint32_t ipmx_stream_id,
    const IPMXSenderReportState& report_state, uint32_t rtp_timestamp, uint64_t receiver_timestamp,
    bool need_update_stats)
{
    bool is_first_update = (stats.frames_received == 0);

    last_frame.tx_ntp_timestamp = report_state.last_report.ntp_ts;
    last_frame.tx_rtp_timestamp = rtp_timestamp;
    last_frame.rx_timestamp = receiver_timestamp;
    last_frame.clock_diff = receiver_timestamp - report_state.last_report.ntp_ts;
    stats.frames_received++;

    if (is_first_update) {
        first_frame = last_frame;
        prev_stats_frame = last_frame;
    }

    if ((static_cast<int64_t>(receiver_timestamp - report_state.stats.last_report_rx_ts) >
        stats.sender_report_to_data_max) || is_first_update) {
        stats.sender_report_to_data_max =
            static_cast<int64_t>(receiver_timestamp - report_state.stats.last_report_rx_ts);
    }

    if ((static_cast<int64_t>(receiver_timestamp - report_state.stats.last_report_rx_ts) <
        stats.sender_report_to_data_min) ||
        is_first_update) {
        stats.sender_report_to_data_min =
            static_cast<int64_t>(receiver_timestamp - report_state.stats.last_report_rx_ts);
    }

    if (need_update_stats) {
        int64_t delay_change = last_frame.clock_diff - prev_stats_frame.clock_diff;
        int64_t time_passed = last_frame.rx_timestamp - prev_stats_frame.rx_timestamp;
        stats.clock_drift_last_period = delay_change;
        stats.clock_ratio_ppm_last_period = (delay_change * 1.0e6) / time_passed;
        int64_t delay_change_since_start = last_frame.clock_diff - first_frame.clock_diff;
        int64_t time_passed_since_start = last_frame.rx_timestamp - first_frame.rx_timestamp;
        stats.clock_drift_since_start = delay_change_since_start;
        stats.clock_ratio_ppm_since_start = (delay_change_since_start * 1.0e6) / time_passed_since_start;

        prev_stats_frame = last_frame;
    }
}

void IPMXStreamTimelineTracker::notify_data_frame(uint32_t rtp_timestamp, uint64_t receiver_timestamp)
{
    IPMXSenderReportState report_state;
    ReturnStatus status = m_report_queue.retrieve_by_tag(rtp_timestamp, report_state);
    if (status != ReturnStatus::success) {
        std::cerr << "IPMX Stream " << m_ipmx_stream_id << ": no Sender Report available" << std::endl;
        return;
    }
    bool need_update_stats = static_cast<int64_t>(m_local_clock_state.last_frame.rx_timestamp -
                             m_local_clock_state.prev_stats_frame.rx_timestamp) >=
                             (m_stats_update_interval_ms - 1) *
                             std::chrono::nanoseconds{ std::chrono::milliseconds{ 1 } }.count();;

    m_local_clock_state.update(m_ipmx_stream_id, report_state, rtp_timestamp, receiver_timestamp, need_update_stats);
    m_shared_clock_state.update(m_local_clock_state);

    if (need_update_stats) {
        m_new_clock_stats_available.store(true);
    }
}

void IPMXStreamTimelineTracker::print_clock_new_info(std::ostream& out)
{
    IPMXDataClockState cs = m_shared_clock_state.read();
    auto old_prec = out.precision(3);
    out << "\n-----------------------------------------------------\n"
        << "IPMX Stream " << m_ipmx_stream_id << "\n"
        << "sender clock   " << ptp_to_string(cs.last_frame.tx_ntp_timestamp) << "\n"
        << "receiver clock " << ptp_to_string(cs.last_frame.rx_timestamp) << "\n"
        << "clock delta " << cs.last_frame.clock_diff << "ns\n"
        << "clock drift (last)  " << cs.stats.clock_drift_last_period << "ns, "
        << cs.stats.clock_ratio_ppm_last_period << "ppm\n"
        << "clock drift (total) " << cs.stats.clock_drift_since_start << "ns, "
        << cs.stats.clock_ratio_ppm_since_start << "ppm\n"
        << "Sender Report to Data delay min " << cs.stats.sender_report_to_data_min / 1000 << "us, max "
        << cs.stats.sender_report_to_data_max / 1000 << "us" << std::endl;
    out.precision(old_prec);
}

void IPMXStreamTimelineTracker::set_statistics_report_interval(uint32_t interval_ms)
{
    m_stats_update_interval_ms = interval_ms;
}

void IPMXStreamTimelineTracker::print_updates()
{
    if (m_new_report_version_available.exchange(false)) {
        std::ostringstream os;
        print_stream_new_info(os);
        std::cout << os.str();
    }
    if (m_new_clock_stats_available.exchange(false)) {
        std::ostringstream os;
        print_clock_new_info(os);
        std::cout << os.str();
    }
}
