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

#ifndef RDK_APPS_RMAX_MEDIA_PROBE_MEDIA_MONITOR_H_
#define RDK_APPS_RMAX_MEDIA_PROBE_MEDIA_MONITOR_H_

#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "rdk/apps/rmax_receiver_base.h"

namespace rivermax
{
namespace dev_kit
{
namespace apps
{
namespace rmax_media_probe
{

/**
 * @brief: Media component identifiers.
 */
 enum MEDIA_COMPONENT_INDEX {
    MEDIA_COMPONENT_VIDEO = 0,
    MEDIA_COMPONENT_ALPHA = 1,
    NUM_OF_MEDIA_COMPONENTS = 2
};

/**
 * @brief: Event structure containing information about a newly detected frame.
 */
struct NewFrameEvent {
    size_t component_index;
    const IReceiveStream& stream;
    uint64_t receive_ts;
    uint32_t rtp_ts;
    uint32_t rtp_seq_num;
    float media_delay_usec;
};

/**
 * @brief: Callback function type for new frame events.
 */
using OnNewFrameCallback = std::function<void (const NewFrameEvent& event)>;

/**
 * @brief: Structure representing a media component with timing and sequence information.
 */
struct MediaComponent {
    uint64_t receive_ts;
    uint32_t rtp_ts;
    bool is_rtp_ts_valid;
    uint32_t rtp_seq_num;
    bool is_extended_seq_num;
    size_t packet_count;
};

/**
 * @brief: Media monitor for tracking and synchronizing multiple media components.
 *
 * This class monitors multiple media components and tracks their synchronization
 * by matching RTP timestamps across components. It provides statistics about
 * frame matching, mismatches, and order errors.
 */
class MediaMonitor {
public:
    /**
     * @brief: MediaMonitor constructor.
     *
     * @param [in] id: Unique identifier for this media monitor.
     * @param [in] num_of_components: Number of media components to monitor.
     */
    MediaMonitor(size_t id, size_t num_of_components) : m_id(id), m_components(num_of_components) {};
    /**
     * @brief: MediaMonitor destructor.
     */
    ~MediaMonitor() = default;
    /**
     * @brief: Handle a new frame event from a component.
     *
     * @param [in] event: New frame event containing component and timing information.
     */
    void on_new_frame(const NewFrameEvent& event);
    /**
     * @brief: Print current statistics and reset counters.
     *
     * @param [out] os: Output stream to write statistics to.
     */
    void print_and_reset_stats(std::ostream& os);
protected:
    /**
     * @brief: Reset RTP timestamp matching state for all components.
     */
    void reset_rtp_matching();
    /**
     * @brief: Restart RTP timestamp matching for a specific component.
     *
     * @param [in] component_index: Index of the component to restart matching for.
     * @param [in] rtp_ts: RTP timestamp to use as the new reference.
     */
    void restart_rtp_matching(size_t component_index, uint32_t rtp_ts);
    /**
     * @brief: Match RTP timestamps across components for synchronization.
     *
     * @param [in] component_index: Index of the component with the new timestamp.
     * @param [in] stream_id: Stream identifier.
     * @param [in] rtp_ts: RTP timestamp to match.
     */
    void match_rtp_timestamps(size_t component_index, uint32_t stream_id, uint32_t rtp_ts);
    /**
     * @brief: Print current statistics without resetting counters.
     *
     * @param [out] os: Output stream to write statistics to.
     */
    void print_stats(std::ostream& os) const;
    /**
     * @brief: Reset all statistics counters to zero.
     */
    void reset_stats();
    /* Mutex for protecting shared state */
    std::mutex m_mtx;
    size_t m_id;
    std::vector<MediaComponent> m_components;
    uint64_t m_matched_frames;
    uint64_t m_mismatches;
    uint64_t m_order_errors;
};

} // namespace rmax_media_probe
} // namespace apps
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_APPS_RMAX_MEDIA_PROBE_MEDIA_MONITOR_H_ */
