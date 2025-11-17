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

#ifndef RDK_APPS_MEDIA_PROBE_MEDIA_MONITOR_H_
#define RDK_APPS_MEDIA_PROBE_MEDIA_MONITOR_H_

#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <functional>
#include <map>

#include "rdk/apps/receiver_base_app.h"
#include "rdk/apps/media_probe/stream_monitor.h"


namespace rivermax
{
namespace dev_kit
{
namespace apps
{
namespace media_probe
{

/**
 * @brief: Structure representing a media component with timing and sequence information.
 */
struct MediaComponent
{
    uint64_t receive_ts = 0;
    uint32_t rtp_ts = 0;
    bool is_rtp_ts_valid = false;
    bool is_rtp_ts_pending = false;
    uint32_t rtp_seq_num = 0;
};

/**
 * @brief: Media monitor for tracking and synchronizing multiple media components.
 *
 * This class monitors multiple media components and tracks their synchronization
 * by matching RTP timestamps across components. It provides statistics about
 * frame matching, mismatches, and order errors.
 */
class MediaMonitor
{
protected:
    std::mutex m_mutex;
    size_t m_id;
    std::map<MediaComponentId, MediaComponent> m_components;
    std::map<MediaComponentId, std::reference_wrapper<StreamMonitor>> m_stream_monitors;
    uint64_t m_matched_frames;
    uint64_t m_mismatches;
public:
    /**
     * @brief: MediaMonitor constructor.
     *
     * @param [in] id: Unique identifier for this media monitor.
     * @param [in] num_of_components: Number of media components to monitor.
     */
    MediaMonitor(size_t id) : m_id(id), m_matched_frames(0), m_mismatches(0) {};
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
     * @param [out] out: Output stream to write statistics to.
     */
    void print_and_reset_stats(std::ostream& out);
    /**
     * @brief: Register a stream monitor with the media monitor.
     *
     * @param [in] stream_monitor: Reference to the stream monitor to register.
     */
     void add_stream_monitor(StreamMonitor& stream_monitor);
protected:
    /**
     * @brief: Reset RTP timestamp matching state for all components.
     */
    void reset_rtp_matching();
    /**
     * @brief: Restart RTP timestamp matching for a specific component.
     *
     * @param [in] component_id: Id of the component to restart matching for.
     * @param [in] rtp_ts: RTP timestamp to use as the new reference.
     */
    void restart_rtp_matching(MediaComponentId component_id, uint32_t rtp_ts);
    /**
     * @brief: Match RTP timestamps across components for synchronization.
     *
     * @param [in] component_id: Id of the component with the new timestamp.
     * @param [in] stream_id: Stream identifier.
     * @param [in] rtp_ts: RTP timestamp to match.
     */
    void match_rtp_timestamps(MediaComponentId component_id, uint32_t stream_id, uint32_t rtp_ts);
    /**
     * @brief: Print current statistics without resetting counters.
     *
     * @param [out] out: Output stream to write statistics to.
     */
    void print_stats(std::ostream& out) const;
    /**
     * @brief: Reset all statistics counters to zero.
     */
    void reset_stats();
};

} // namespace media_probe
} // namespace apps
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_APPS_MEDIA_PROBE_MEDIA_MONITOR_H_ */
