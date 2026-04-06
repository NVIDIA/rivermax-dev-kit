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

#include "rdk/services/utils/counter_math.h"
#include "rdk/apps/media_probe/media_monitor.h"

using namespace rdk::apps;

void MediaMonitor::reset_rtp_matching()
{
    for (auto& component_pair : m_components) {
        component_pair.second.is_rtp_ts_pending = false;
    }
}

void MediaMonitor::restart_rtp_matching(MediaComponentId component_id, uint32_t rtp_ts)
{
    reset_rtp_matching();
    m_components[component_id].is_rtp_ts_pending = true;
    m_components[component_id].rtp_ts = rtp_ts;
}

/**
 * @brief: Match RTP timestamps across media components for synchronization.
 *
 * This algorithm verifies that the sequences of RTP timestamps in the incoming
 * media component streams are equal. This rule is applicable to video and alpha/key streams
 * as well as for ancillary streams representing the same media content.
 *
 * The algorithm works in the assumptions that the incoming data is buffered in chunks
 * in the amount not higher than one frame: when two streams are sent synchronously
 * (both have the same FPS and both comply to ST2110-21, one stream can never be fetched
 * with more than one frame ahead than another stream.
 * The idea is the following:
 * Each stream has a flag is_rtp_ts_valid, meaning that a stream has a new frame that
 * is pending matching with other components. The flag is cleared when all streams
 * get a frame with the same timestamp (matching succeeded, a new matching cycle starts),
 * or when another stream receives a frame with non-matching timestamp (matching failed,
 * a new matching cycle starts).
 * When a new frame is detected in one stream, its RTP timestamp is compared to the
 * previous RTP timestamps of this stream and with the last timestamps of other streams.
 */
void MediaMonitor::match_rtp_timestamps(MediaComponentId component_id, uint32_t stream_id, uint32_t rtp_ts)
{
    m_components[component_id].is_rtp_ts_valid = true;
    m_components[component_id].rtp_ts = rtp_ts;

    size_t matching_count = 0;
    for (auto& component_pair : m_components) {
        auto& component = component_pair.second;
        if (component_pair.first != component_id) {
            /* this is the another stream  */
            if (component.is_rtp_ts_pending) {
                if (is_counter_before(rtp_ts, component.rtp_ts)) {
                    /* new timestamp is lower that the pending timestamp in another stream, matching failed */
                    m_mismatches++;
                    reset_rtp_matching();
                    return;
                } else if (is_counter_before(component.rtp_ts, rtp_ts)) {
                    /* new timestamp is higher that the pending timestamp in another stream, matching failed */
                    m_mismatches++;
                    /* keep the new timestamp, maybe this is the next frame */
                    restart_rtp_matching(component_id, rtp_ts);
                    return;
                }
                /* new timestamp matches the pending timestamp in another stream, matching succeeded */
                matching_count++;
            }
        }
    }
    /* no mismatches found */
    if (matching_count == m_components.size() - 1) {
        /* all streams are matched */
        m_matched_frames++;
        reset_rtp_matching();
    } else {
        /* some streams don't have a pending timestamp, keep the new timestamp */
        m_components[component_id].is_rtp_ts_pending = true;
    }
}

void MediaMonitor::on_new_frame(const NewFrameEvent& event)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_components.count(event.component_id) == 0) {
        std::cerr << "Invalid media component Id in new frame event" << std::endl;
        return;
    }

    match_rtp_timestamps(event.component_id, event.stream.get_id(), event.rtp_ts);

    m_components[event.component_id].receive_ts = event.receive_ts;
    m_components[event.component_id].rtp_seq_num = event.rtp_seq_num;
}

void MediaMonitor::print_stats(std::ostream& out) const
{
    std::stringstream oss;
    oss << "Media id: " << m_id
        << " video/alpha frame matches: " << m_matched_frames
        << " mismatches: " << m_mismatches
        << std::endl;
    out << oss.str();
}

void MediaMonitor::reset_stats()
{
    m_matched_frames = 0;
    m_mismatches = 0;
}

void MediaMonitor::print_and_reset_stats(std::ostream& out)
{
    std::stringstream oss;
    std::unique_lock<std::mutex> lock(m_mutex);
    oss << "--------------------------------" << std::endl;
    print_stats(oss);
    for (auto& stream_monitor_pair : m_stream_monitors) {
        stream_monitor_pair.second.get().print_and_reset_stats(oss);
    }
    out << oss.str();
    reset_stats();
}

void MediaMonitor::add_stream_monitor(StreamMonitor& stream_monitor)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    stream_monitor.set_on_new_frame_callback(std::bind(&MediaMonitor::on_new_frame, this, std::placeholders::_1));
    m_stream_monitors.emplace(stream_monitor.get_component_id(), std::ref(stream_monitor));
    MediaComponent component;
    m_components.emplace(stream_monitor.get_component_id(), component);
}
