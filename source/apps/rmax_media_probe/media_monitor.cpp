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

#include "rdk/apps/rmax_media_probe/media_monitor.h"

using namespace rivermax::dev_kit::apps::rmax_media_probe;

/**
 * @brief: A helper function to compare uint32_t values (e.g. RTP timestamps) with care to wrap-around.
 *
 * A is considered before B if there is a X: 0 <= X <= 0x7fffffff such that B = (A + X) mod 2^32.
 *
 * @param [in] a: The first argument of "is before" comparison.
 * @param [in] b: The second argument of "is before" comparison.
 *
 * @return: True if a is before b.
 */
 static inline bool is_before(uint32_t a, uint32_t b) {
    return (((a & 0x80000000) == (b & 0x80000000)) && (a < b)) ||
           (((a & 0x80000000) != (b & 0x80000000)) && ((a & 0x7fffffff) > (b & 0x7fffffff)));
}

void MediaMonitor::reset_rtp_matching()
{
    for (size_t i = 0; i < m_components.size(); ++i) {
        m_components[i].is_rtp_ts_pending = false;
    }
}

void MediaMonitor::restart_rtp_matching(size_t component_index, uint32_t rtp_ts)
{
    reset_rtp_matching();
    m_components[component_index].is_rtp_ts_pending = true;
    m_components[component_index].rtp_ts = rtp_ts;
}

/**
 * @brief: Match RTP timestamps across mediacomponents for synchronization.
 *
 * The algorithm works in the assumptions that the incoming data is buffered in chunks
 * in the amount not higher than one frame: when two streams are sent synchronously
 * (both have the same FPS and both comply to ST2110-21, one stream can never be fetched
 * with more than one frame ahead than another stream.
 * The idea is the following:
 * Each stream has a flag is_rtp_ts_valid, meaning that a stream has a new frame that
 * is pending matching with other components. The flag is cleared when all streams
 * get a frame with the same timestamp (matching succeeded, a new matchinch cycle starts),
 * or when another stream receives a a frame with non-matching timestamp (matching failed,
 * a new matching cycle starts).
 * When a new frame is detected in one stream, its RTP timestamp is compared to the
 * previous RTP timestamps of this stream and with the last timestamps of other streams.
 */
void MediaMonitor::match_rtp_timestamps(size_t component_index, uint32_t stream_id, uint32_t rtp_ts)
{
    if (likely(m_components[component_index].is_rtp_ts_valid)) {
        if ((rtp_ts == m_components[component_index].rtp_ts) || is_before(rtp_ts, m_components[component_index].rtp_ts)) {
            m_order_errors++;
            reset_rtp_matching();
            return;
        }
    }
    m_components[component_index].is_rtp_ts_valid = true;
    m_components[component_index].rtp_ts = rtp_ts;

    size_t matching_count = 0;
    for (size_t i = 0; i < m_components.size(); ++i) {
        if (i != component_index) {
            /* this is the another stream  */
            if (m_components[i].is_rtp_ts_pending) {
                if (is_before(rtp_ts, m_components[i].rtp_ts)) {
                    /* new timestamp is lower that the pending timestamp in another stream, matching failed */
                    m_mismatches++;
                    reset_rtp_matching();
                    return;
                } else if (is_before(m_components[i].rtp_ts, rtp_ts)) {
                    /* new timestamp is higher that the pending timestamp in another stream, matching failed */
                    m_mismatches++;
                    /* keep the new timestamp, maybe this is the next frame */
                    restart_rtp_matching(component_index, rtp_ts);
                    return;
                }
                /* new timestamp is matches the pending timestamp in another stream, matching succeeded */
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
        m_components[component_index].is_rtp_ts_pending = true;
    }
}

void MediaMonitor::on_new_frame(const NewFrameEvent& event)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    if (event.component_index >= m_components.size()) {
        std::cerr << "component index out of range" << std::endl;
        return;
    }

    match_rtp_timestamps(event.component_index, event.stream.get_id(), event.rtp_ts);

    m_components[event.component_index].receive_ts = event.receive_ts;
    m_components[event.component_index].rtp_seq_num = event.rtp_seq_num;
}

void MediaMonitor::print_stats(std::ostream& os) const
{
    os << "Media id: " << m_id
       << " matched frames: " << m_matched_frames
       << " mismatches: " << m_mismatches
       << " order errors: " << m_order_errors
       << std::endl;
}

void MediaMonitor::reset_stats()
{
    m_matched_frames = 0;
    m_mismatches = 0;
    m_order_errors = 0;
}

void MediaMonitor::print_and_reset_stats(std::ostream& os)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    print_stats(os);
    reset_stats();
}

