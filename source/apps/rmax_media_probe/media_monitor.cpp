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
        m_components[i].is_rtp_ts_valid = false;
    }
}

void MediaMonitor::restart_rtp_matching(size_t component_index, uint32_t rtp_ts)
{
    reset_rtp_matching();
    m_components[component_index].is_rtp_ts_valid = true;
    m_components[component_index].rtp_ts = rtp_ts;
}

void MediaMonitor::match_rtp_timestamps(size_t component_index, uint32_t stream_id, uint32_t rtp_ts)
{
    size_t matching_count = 0;
    for (size_t i = 0; i < m_components.size(); ++i) {
        if (m_components[i].is_rtp_ts_valid) {
            if (is_before(rtp_ts, m_components[i].rtp_ts)) {
                m_order_errors++;
                reset_rtp_matching();
                return;
            }
            if (is_before(m_components[i].rtp_ts, rtp_ts)) {
                m_mismatches++;
                restart_rtp_matching(component_index, rtp_ts);
                return;
            }
            matching_count++;
        }
        if (i == component_index) {
            if (m_components[i].is_rtp_ts_valid) {
                m_order_errors++;
                reset_rtp_matching();
                return;
            } else {
                m_components[i].is_rtp_ts_valid = true;
                m_components[i].rtp_ts = rtp_ts;
                matching_count++;
            }
        }
    }
    if (matching_count == m_components.size()) {
        m_matched_frames++;
        reset_rtp_matching();
        return;
    }
}

void MediaMonitor::on_new_frame(const NewFrameEvent& event)
{
    std::unique_lock<std::mutex> lock(m_mtx);
    if (event.component_index >= m_components.size()) {
        std::cout << "component index out of range" << std::endl;
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
    std::unique_lock<std::mutex> lock(m_mtx);
    print_stats(os);
    reset_stats();
}

