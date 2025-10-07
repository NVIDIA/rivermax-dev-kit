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

#ifndef RDK_APPS_RMAX_XSTREAM_MEDIA_SENDER_RMAX_XSTREAM_MEDIA_SENDER_H_
#define RDK_APPS_RMAX_XSTREAM_MEDIA_SENDER_RMAX_XSTREAM_MEDIA_SENDER_H_

#include <memory>
#include <functional>
#include <unordered_map>

#include "rdk/apps/rmax_base_app.h"
#include "rdk/services/media/media_settings.h"
#include "rdk/services/media/app_media_settings.h"

using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
{
namespace apps
{
namespace rmax_xstream_media_sender
{
/**
 * @brief: Configuration settings for Rivermax Media Sender.
 */
struct MediaSenderSettings : AppSettings
{
public:
    static constexpr uint32_t DEFAULT_NUM_OF_PACKETS_IN_CHUNK_FHD = 16;
    static constexpr uint32_t DEFAULT_NUM_OF_PACKETS_IN_CHUNK_UHD = 32;
    static constexpr uint32_t DEFAULT_FRAME_FIELDS_IN_MEM_BLOCK = 10;
    void init_default_values() override;
    std::unordered_set<SMPTEStandard> enabled_smpte_standards;
    std::vector<std::unique_ptr<MediaSettings>> smpte_standard_configs;
    std::vector<std::pair<const MediaSettings&, size_t>> smpte_standard_to_nodes;
};

/**
 * @brief: Validator for Rivermax Media Sender settings.
 */
class MediaSenderSettingsValidator : public ISettingsValidator<MediaSenderSettings>
{
public:
     ReturnStatus validate(const std::shared_ptr<MediaSenderSettings>& settings) const override;
};

/**
 * @brief: CLI settings Builder for Rivermax Media Sender.
 */
class MediaSenderCLISettingsBuilder : public CLISettingsBuilder<MediaSenderSettings>
{
public:
    /**
     * @brief: MediaSenderCLISettingsBuilder constructor.
     *
     * @param [in] argc: Number of CLI arguments.
     * @param [in] argv: CLI arguments strings array.
     * @param [in] app_description: Application description string for the CLI usage.
     * @param [in] app_examples: Application examples string for the CLI usage.
     */
    MediaSenderCLISettingsBuilder(int argc, const char** argv,
        const std::string& app_description,
        const std::string& app_examples,
        std::shared_ptr<ISettingsValidator<MediaSenderSettings>> validator) :
        CLISettingsBuilder<MediaSenderSettings>(argc, argv, app_description, app_examples, std::move(validator)) {}
    virtual ~MediaSenderCLISettingsBuilder() = default;
protected:
    ReturnStatus add_cli_options(std::shared_ptr<MediaSenderSettings>& settings) override;
};

using MediaSenderExternalSettingsBuilder = ExternalSettingsBuilder<MediaSenderSettings>;
/**
 * @brief: Media Sender application.
 *
 * This is an example of usage application for Rivermax media TX API.
 */
class MediaSenderApp : public RmaxBaseApp
{
private:
    /* Settings builder pointer */
    std::shared_ptr<ISettingsBuilder<MediaSenderSettings>> m_settings_builder;
    /* Application settings pointer */
    std::shared_ptr<MediaSenderSettings> m_media_sender_settings;
    /* Sender objects container */
    std::vector<std::shared_ptr<MediaSenderIONode>> m_senders;
    /* NIC device interface */
    rmx_device_iface m_device_interface;
    /* Number of paths per stream */
    size_t m_num_paths_per_stream = 1;
    /* Network send flows */
    std::vector<TwoTupleFlow> m_flows;
    /**
     * @brief: Media type configuration function map.
     *
     * This static map contains functions for configuring different SMPTE standards.
     * Each SMPTEStandard enum value maps to a function that configures that specific media type.
     */
    static const std::unordered_map<SMPTEStandard, std::function<ReturnStatus(MediaSenderApp*)>> s_smpte_standard_config_map;
public:
    /**
     * @brief: MediaSenderApp class constructor.
     *
     * @param [in] settings_builder: Settings builder pointer.
     */
    MediaSenderApp(std::shared_ptr<ISettingsBuilder<MediaSenderSettings>> settings_builder);
    virtual ~MediaSenderApp() = default;
    ReturnStatus run() override;
    /**
     * @brief: Initializes SMPTE standards configuration.
     *
     * This method is responsible for initializing the configuration of different SMPTE standards
     * that will be used by the sender application.
     *
     * @return: Status of the operation.
     */
    ReturnStatus initialize_smpte_standards();
    ReturnStatus initialize() override;
    /**
     * @brief: Sets the frame provider for the specified stream index.
     *
     * @param [in] stream_index: Stream index.
     * @param [in] frame_provider: Framer provider pointer.
     * @param [in] smpte_standard: SMPTE standard.
     * @param [in] contains_payload: Flag indicating whether the frame provider contains payload.
     *
     * @return: Status of the operation.
     */
    ReturnStatus set_frame_provider(size_t stream_index, std::shared_ptr<IFrameProvider> frame_provider,
        SMPTEStandard smpte_standard, bool contains_payload = true);
private:
    ReturnStatus initialize_app_settings() final;
    ReturnStatus post_load_settings() final;
    ReturnStatus initialize_memory_strategy() override;
    ReturnStatus set_rivermax_clock() override;
    ReturnStatus initialize_connection_parameters() final;

    /**
     * @brief: Configures video types processing.
     *
     * This method is responsible for configuring video media types processing
     * for the sender application.
     *
     * @return: Status of the operation.
     */
    ReturnStatus configure_video_types();
    /**
     * @brief: Distributes streams across threads.
     *
     * This method is responsible for distributing streams across threads based on
     * the number of threads and minimum streams per thread requirements.
     *
     * @param [in] num_of_threads: Number of threads to distribute streams across.
     * @param [in] min_streams_per_thread: Minimum number of streams per thread.
     * @param [in] media_settings: Media settings configuration.
     */
    void distribute_streams_across_threads(size_t num_of_threads, size_t min_streams_per_thread, const MediaSettings& media_settings);
    /**
     * @brief: Initializes network send flows.
     *
     * This method is responsible to initialize the send flows will be
     * used in the application. Those flows will be distributed in
     * @ref MediaSenderApp::distribute_work_for_threads
     * to the streams will be used in the application.
     * The application supports unicast and multicast UDPv4 send flows.
     */
    void configure_network_flows();
    /**
     * @brief: Configures processing of enabled SMPTE standards.
     *
     * This method is responsible to configure processing of enabled SMPTE standards
     * for the sender application.
     */
    ReturnStatus configure_smpte_standards_processing();
    /**
     * @brief: Initializes sender threads.
     *
     * This method is responsible to initialize @ref MediaSenderIONode objects to work.
     * It will initiate objects with the relevant parameters.
     * The objects initialized in this method, will be the contexts to the std::thread objects
     * will run in @ref RmaxBaseApp::run_threads method.
     */
    void initialize_sender_threads();
    /**
     * @brief: Returns current time in nanoseconds.
     *
     * This method uses @ref get_rivermax_ptp_time_ns to return the current PTP time.
     * @note: PTP4l must be running on the system for time to be valid.
     *
     * @return: Current time in nanoseconds.
     */
    static uint64_t get_time_ns(void* context = nullptr);
    /**
     * @brief: Sets internal frame providers.
     *
     * This method is responsible to set internal (default) frame providers for
     * the streams. The internal frame providers are used to generate frames that
     * will be set by BufferWriters as a payload. User will be able to set an
     * external frame provider by @ref MediaSenderApp::set_frame_provider.
     *
     * @return: Status of the operation.
     */
    ReturnStatus set_internal_frame_providers();
};

} // namespace rmax_xstream_media_sender
} // namespace apps
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_APPS_RMAX_XSTREAM_MEDIA_SENDER_RMAX_XSTREAM_MEDIA_SENDER_H_ */
