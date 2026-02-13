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

#ifndef RDK_APPS_MEDIA_SENDER_MEDIA_SENDER_H_
#define RDK_APPS_MEDIA_SENDER_MEDIA_SENDER_H_

#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <unordered_map>

#include "rdk/apps/base_app.h"
#include "rdk/services/media/app_media_settings.h"
#include "rdk/services/media/media_settings.h"

using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
{
namespace apps
{
namespace media_sender
{
/**
 * @brief: Configuration settings for Rivermax Dev Kit Media Sender.
 */
struct MediaSenderSettings : AppSettings
{
private:
    static inline const std::unordered_map<uint32_t, uint32_t> DEFAULT_PACKETS_IN_CHUNK = {
        { static_cast<uint32_t>(_1080_WIDTH * _1080_HEIGHT), 16 },
        { static_cast<uint32_t>(_2160_WIDTH * _2160_HEIGHT), 32 },
        { static_cast<uint32_t>(_4320_WIDTH * _4320_HEIGHT), 64 },
    };
public:
    static constexpr uint32_t DEFAULT_FRAME_FIELDS_IN_MEM_BLOCK = 10;

    void init_default_values() override;
    std::set<SMPTEStandard> enabled_smpte_standards;
    std::vector<std::unique_ptr<MediaSettings>> smpte_standard_configs;
    std::vector<std::pair<const MediaSettings&, size_t>> smpte_standard_to_nodes;

    /**
     * @brief: Returns the default number of packets in chunk for a given resolution.
     *
     * @param [in] resolution: The resolution to get the default number of packets in chunk for.
     *
     * @return: The default number of packets in chunk.
     */
    static uint32_t get_default_packets_in_chunk(const Resolution& resolution)
    {
        auto it = DEFAULT_PACKETS_IN_CHUNK.find(static_cast<uint32_t>(resolution.width) * resolution.height);
        if (it == DEFAULT_PACKETS_IN_CHUNK.end()) {
            std::cerr << "No default packets in chunk found for resolution: " << resolution.width << "x"
                      << resolution.height << std::endl;
            std::cerr << "Using default packets in chunk for: " << Resolution(_1080_WIDTH, _1080_HEIGHT) << std::endl;
            return DEFAULT_PACKETS_IN_CHUNK.at(static_cast<uint32_t>(_1080_WIDTH * _1080_HEIGHT));
        }
        return it->second;
    }
};

/**
 * @brief: Validator for Rivermax Dev Kit Media Sender settings.
 */
class MediaSenderSettingsValidator : public ISettingsValidator<MediaSenderSettings>
{
public:
    ReturnStatus validate(const MediaSenderSettings& settings) const override;
};

/**
 * @brief: CLI settings Builder for Rivermax Dev Kit Media Sender.
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
     * @param [in] validator: A const reference to the settings validator.
     */
    MediaSenderCLISettingsBuilder(int argc, const char** argv,
        const std::string& app_description,
        const std::string& app_examples,
        const ISettingsValidator<MediaSenderSettings>& validator) :
        CLISettingsBuilder<MediaSenderSettings>(argc, argv, app_description, app_examples, validator) {}
    virtual ~MediaSenderCLISettingsBuilder() = default;
protected:
    ReturnStatus add_cli_options(MediaSenderSettings& settings) override;
};

using MediaSenderUserProvidedSettingsBuilder = UserProvidedSettingsBuilder<MediaSenderSettings>;
/**
 * @brief: Media Sender application.
 *
 * This is an example of usage application for Rivermax Dev Kit Media Sender.
 */
class MediaSenderApp : public BaseApp
{
private:
    /* Settings builder pointer */
    std::unique_ptr<ISettingsBuilder<MediaSenderSettings>> m_settings_builder;
    /* Application settings pointer */
    std::shared_ptr<MediaSenderSettings> m_media_sender_settings;
    /* Sender objects container */
    std::vector<std::shared_ptr<MediaSenderIONode>> m_senders;
    /* Number of paths per stream */
    size_t m_num_paths_per_stream = 1;
    /* Network send flows */
    std::vector<FourTupleFlow> m_flows;
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
    MediaSenderApp(std::unique_ptr<ISettingsBuilder<MediaSenderSettings>> settings_builder);
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
     * @brief: Sets media essence sources for a specific stream.
     *
     * This is the application-level API that forwards to the IO node level.
     * It forwards the request to the appropriate MediaSenderIONode instance based on the stream index.
     * This method configures the media essence sources that supply media data to a stream.
     * Two types of sources can be configured:
     *
     * - **Preload Source**: Pre-fills memory blocks with media data before transmission begins.
     *   This is a one-time operation that prepares data in advance for optimal performance.
     *
     * - **Runtime Source**: Supplies fresh media data dynamically during active transmission.
     *   Called continuously as new media units are available.
     *
     * @par Usage Patterns:
     * 1. **Static Content**: Set only a preload source and disable runtime payload copying
     *    (`runtime_contains_payload = false`) for maximum efficiency when transmitting
     *    the same data repeatedly.
     *
     * 2. **Dynamic Content**: Set only a runtime source when media data changes continuously.
     *
     * 3. **Hybrid Mode**: Set both sources - preload fills memory blocks once before
     *    transmission starts, while runtime supplies new media units to send when they
     *    become available during the transmission loop.
     *
     * @par Default Behavior:
     * Each stream is initialized with @ref NullEssenceSource for both sources by default.
     * @ref NullEssenceSource generates only RTP headers; payload data is not written
     * during the transmission loop. At least one source should be set to a real
     * implementation for meaningful data transmission.
     *
     * @param [in] stream_index: The external stream index to configure.
     * @param [in] smpte_standard: The SMPTE standard for media formatting.
     * @param [in] preload_essence_source: Source for preloading data into memory blocks
     *                                     before transmission. Pass nullptr to preserve the
     *                                     existing preload source (default: nullptr).
     * @param [in] runtime_essence_source: Source for supplying media data during active
     *                                     transmission. Pass nullptr to preserve the existing
     *                                     runtime source (default: nullptr).
     * @param [in] runtime_contains_payload: If true, copies both headers and payload from the
     *                                       runtime source. If false, only constructs RTP
     *                                       headers from the runtime source, leaving payload
     *                                       data untouched (assumes preloaded). Setting to false
     *                                       improves performance when payload is static
     *                                       (default: true).
     *
     * @note: The @p runtime_contains_payload parameter only affects the runtime source's behavior.
     *        The @p preload_essence_source always writes complete data (headers and payload).
     *
     * @return: Status of the operation.
     */
    ReturnStatus set_media_essence_sources(
        size_t stream_index,
        SMPTEStandard smpte_standard,
        std::shared_ptr<IMediaEssenceSource> preload_essence_source = nullptr,
        std::shared_ptr<IMediaEssenceSource> runtime_essence_source = nullptr,
        bool runtime_contains_payload = true);
private:
    ReturnStatus initialize_app_settings() final;
    ReturnStatus post_load_settings() final;
    ReturnStatus initialize_memory_strategy() override;
    ReturnStatus set_rivermax_clock() override;
    /**
     * @brief: Generic helper for configuring media types.
     *
     * This template method handles the common logic for distributing streams across threads
     * and configuring media settings.
     *
     * @tparam SettingsType: The media settings type.
     * @param [in] settings: Media settings object.
     * @param [in] smpte_standard_name: SMPTE standard identifier for logging.
     *
     * @return: Status of the operation.
     */
    template <typename SettingsType>
    ReturnStatus configure_media_type_helper(std::unique_ptr<SettingsType> settings, const std::string& smpte_standard_name);
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
     * @brief: Configures audio types processing.
     *
     * This method is responsible for configuring audio media types processing
     * for the sender application.
     */
    ReturnStatus configure_audio_types();
    /**
     * @brief: Configures ancillary types processing.
     *
     * This method is responsible for configuring ancillary media types processing
     * for the sender application.
     */
    ReturnStatus configure_ancillary_types();
    /**
     * @brief: Initializes network send flows.
     *
     * This method is responsible to initialize the send flows will be
     * used in the application.
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
     * will run in @ref BaseApp::run_threads method.
     *
     * @return: Status of the operation.
     */
    ReturnStatus initialize_sender_threads();
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
     * @brief: Sets internal media essence sources.
     *
     * This method is responsible to set internal (default) media essence sources for
     * the streams. The internal media essence sources are used to generate media units that
     * will be set by BufferWriters as a payload. User will be able to set an
     * external media essence source by @ref MediaSenderApp::set_media_essence_sources.
     *
     * @return: Status of the operation.
     */
    ReturnStatus set_internal_media_essence_sources();
};

} // namespace media_sender
} // namespace apps
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_APPS_MEDIA_SENDER_MEDIA_SENDER_H_ */
