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

#ifndef RDK_APPS_MEDIA_PROBE_MEDIA_PROBE_H_
#define RDK_APPS_MEDIA_PROBE_MEDIA_PROBE_H_

#include <memory>

#include "rdk/apps/receiver_base_app.h"
#include "rdk/apps/media_probe/stream_monitor.h"
#include "rdk/apps/media_probe/media_monitor.h"

namespace rdk
{
namespace apps
{

/**
 * @brief: Configuration settings for Rivermax Dev Kit Media Probe.
 */
struct MediaProbeSettings : AppSettings
{
public:
    static constexpr uint32_t DEFAULT_NUM_OF_PACKETS_IN_CHUNK = 262144;
    /**
     * @brief: Initialize default values for media probe settings.
     */
    void init_default_values() override;

    bool is_video_enabled;
    bool is_alpha_enabled;
};

/**
 * @brief: Validator for Rivermax Dev Kit Media Probe settings.
 */
class MediaProbeSettingsValidator : public ISettingsValidator<MediaProbeSettings>
{
public:
    /**
     * @brief: Validate media probe settings.
     *
     * @param [in] settings: Shared pointer to the settings to validate.
     * @return: Return status indicating validation success or failure.
     */
    ReturnStatus validate(const MediaProbeSettings& settings) const override;
};

/**
 * @brief: CLI settings Builder for Rivermax Dev Kit Media Probe.
 */
class MediaProbeCLISettingsBuilder : public CLISettingsBuilder<MediaProbeSettings>
{
public:
    /**
     * @brief: MediaProbeCLISettingsBuilder constructor.
     *
     * @param [in] argc: Number of CLI arguments.
     * @param [in] argv: CLI arguments strings array.
     * @param [in] app_description: Application description string for the CLI usage.
     * @param [in] app_examples: Application examples string for the CLI usage.
     */
    MediaProbeCLISettingsBuilder(int argc, const char** argv,
        const std::string& app_description,
        const std::string& app_examples,
        const ISettingsValidator<MediaProbeSettings>& validator) :
        CLISettingsBuilder<MediaProbeSettings>(argc, argv, app_description, app_examples, validator) {}
    virtual ~MediaProbeCLISettingsBuilder() = default;
protected:
    /**
     * @brief: Add command line interface options for media probe settings.
     *
     * @param [in,out] settings: Shared pointer to the settings to configure.
     * @return: Return status indicating success or failure.
     */
    ReturnStatus add_cli_options(MediaProbeSettings& settings) override;
};

/**
 * @brief: External settings builder for Rivermax Dev Kit Media Probe.
 */
using MediaProbeUserProvidedSettingsBuilder = UserProvidedSettingsBuilder<MediaProbeSettings>;
/**
 * @brief: Media Probe application.
 *
 * This is an example of application that uses Rivermax Dev Kit to receive the
 * Main Essence (color) and Alpha/Key channels of a video signal.
 * These two components are received as separate ST2110-20 RTP streams.
 * The application tracks the synchronization of these components by matching
 * RTP timestamps across the two streams.
 * The application also tracks:
 *  - the media latency of each component by calculating
 *    the difference between the receive timestamp and the RTP timestamp.
 *  - the number of dropped packets by calculating tracking RTP sequence numbers.
 *  - the number of received frames, packets, bytes.
 * The application is capable of receiving multiple video and alpha/key streams.
 */
class MediaProbeApp : public ReceiverBaseApp
{
private:
    /* Settings builder pointer */
    std::unique_ptr<ISettingsBuilder<MediaProbeSettings>> m_settings_builder;
    /* Application settings pointer */
    std::shared_ptr<MediaProbeSettings> m_media_probe_settings;
    /* Network recv flows */
    std::vector<ReceiveFlow> m_video_flows;
    std::vector<ReceiveFlow> m_alpha_flows;
    /* Frame extractors */
    std::vector<std::unique_ptr<StreamMonitor>> m_stream_monitors;
    /* Media monitors */
    std::vector<std::unique_ptr<MediaMonitor>> m_media_monitors;
public:
    /**
     * @brief: MediaProbeApp class constructor.
     *
     * @param [in] settings_builder: Settings builder pointer.
     */
    MediaProbeApp(std::unique_ptr<ISettingsBuilder<MediaProbeSettings>> settings_builder);
    /**
     * @brief: MediaProbeApp class destructor.
     */
    virtual ~MediaProbeApp() = default;
    /**
     * @brief: Get RTP streams total statistics.
     *
     * This function hides the base class function and provides default template arguments.
     *
     * @return: A vector of @ref RXStatistics.
     */
    std::vector<RXStatistics> get_streams_total_statistics() const {
        return ReceiverBaseApp::get_streams_total_statistics<RXStatistics, AppRTPReceiveStream>();
    }
private:
    /**
     * @brief: Initialize application-specific settings.
     *
     * @return: Return status indicating success or failure.
     */
    ReturnStatus initialize_app_settings() final;
    /**
     * @brief: Set the Rivermax clock for timing operations.
     *
     * @return: Return status indicating success or failure.
     */
    ReturnStatus set_rivermax_clock() final;
    /**
     * @brief: Configure network flows for video and alpha channels.
     */
    void configure_network_flows() final;
    /**
     * @brief: Initialize media probe node streams for a specific component.
     *
     * @param [in,out] node: RTP receiver IO node to configure.
     * @param [in] start_media_index: Starting media index for the flows.
     * @param [in] flows: Vector of receive flows to configure.
     * @param [in] component_index: Index of the component being configured.
     */
    void initialize_media_probe_node_streams(RTPReceiverIONode& node, size_t start_id,
        const std::vector<ReceiveFlow>& flows, MediaComponentId component_index);
    /**
     * @brief: Initialize receive IO nodes for the application.
     */
    void initialize_receive_io_nodes() final;
    /**
     * @brief: Run receiver threads for processing incoming data.
     */
    void run_receiver_threads() final;
    /**
     * @brief: Initialize receiver IO nodes for a specific media component.
     *
     * @param [in,out] receiver_index: Reference to the current receiver index, will be incremented.
     * @param [in] component_index: index of the media component (VIDEO or ALPHA).
     * @param [in] flows: Vector of receive flows for this component.
     */
    void initialize_component_receivers(size_t& receiver_index, MediaComponentId component_index, const std::vector<ReceiveFlow>& flows);
    /**
     * @brief: Limit the receive chunk size.
     *
     * Limit the receive chunk size below one entire frame to ensure correct frame matching.
     */
    static constexpr size_t RECEIVE_CHUNK_SIZE_LIMIT = 1024;
};

} // namespace apps
} // namespace rdk

#endif /* RDK_APPS_MEDIA_PROBE_MEDIA_PROBE_H_ */
