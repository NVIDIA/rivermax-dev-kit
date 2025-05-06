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

#include "rdk/apps/rmax_base_app.h"

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
    void init_default_values() override;
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
public:
    /**
     * @brief: MediaSenderApp class constructor.
     *
     * @param [in] settings_builder: Settings builder pointer.
     */
    MediaSenderApp(std::shared_ptr<ISettingsBuilder<MediaSenderSettings>> settings_builder);
    virtual ~MediaSenderApp() = default;
    ReturnStatus run() override;
    ReturnStatus initialize() override;
    /**
     * @brief: Sets the frame provider for the specified stream index.
     *
     * @param [in] stream_index: Stream index.
     * @param [in] frame_provider: Framer provider pointer.
     * @param [in] media_type: Media type.
     * @param [in] contains_payload: Flag indicating whether the frame provider contains payload.
     *
     * @return: Status of the operation.
     */
    ReturnStatus set_frame_provider(size_t stream_index, std::shared_ptr<IFrameProvider> frame_provider,
        MediaType media_type = MediaType::Video, bool contains_payload = true);
private:
    ReturnStatus initialize_app_settings() final;
    ReturnStatus post_load_settings() final;
    ReturnStatus initialize_memory_strategy() override;
    ReturnStatus set_rivermax_clock() override;
    ReturnStatus initialize_connection_parameters() final;
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
     * @brief: Distributes work for threads.
     *
     * This method is responsible to distribute work to threads, by
     * distributing number of streams per sender thread uniformly.
     * In future development, this can be extended to different
     * streams per thread distribution policies.
     */
    void distribute_work_for_threads();
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
