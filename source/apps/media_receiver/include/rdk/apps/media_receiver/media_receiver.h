/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef RDK_APPS_MEDIA_RECEIVER_MEDIA_RECEIVER_H_
#define RDK_APPS_MEDIA_RECEIVER_MEDIA_RECEIVER_H_

#include <functional>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "rdk/apps/receiver_base_app.h"
#include "rdk/services/media/app_media_settings.h"
#include "rdk/services/media/media_essence_sink.h"
#include "rdk/services/media/media_settings.h"
#include "rdk/services/media/media_unit_pool.h"
// TODO: #include "rdk/services/ulp_packet_buffer/readers/rtp_media_packet_buffer_reader.h"

using namespace rdk::core;
using namespace rdk::io_node;
using namespace rdk::services;

namespace rdk
{
namespace apps
{

/**
 * @brief: Data consumer that reconstructs media units from RTP packets.
 *
 * Implements @ref IReceiveDataConsumer to bridge chunk-level packet reception with
 * media unit-level delivery.
 *
 * TODO: Add std::unique_ptr<IULPPacketBufferReader> member and accept it in the constructor.
 */
class MediaReconstructionDataConsumer : public IReceiveDataConsumer
{
public:
    /**
     * @brief: Constructor for MediaReconstructionDataConsumer.
     *
     * @param [in] buffer_reader: Buffer reader for packet parsing and reconstruction.
     * @param [in] sink: Sink to deliver completed media units.
     * @param [in] pool: Pool for allocating media unit buffers.
     */
    MediaReconstructionDataConsumer(
        // TODO: std::unique_ptr<IULPPacketBufferReader> buffer_reader,
        std::shared_ptr<IMediaEssenceSink> sink,
        std::shared_ptr<MediaUnitPool> pool);

    virtual ~MediaReconstructionDataConsumer() = default;

    /**
     * @brief: Processes packets from chunk, reconstructs media units, delivers to sink.
     *
     * For each chunk:
     * 1. Acquires output MediaUnit from pool if needed
     * 2. Calls buffer reader to parse packets and fill MediaUnit
     * 3. When media unit is complete, delivers to sink
     * 4. Sets the next media unit for the buffer reader
     *
     * @param [in] chunk: Reference to the @ref ReceiveChunk containing the packets.
     * @param [in] stream: Reference to the @ref IReceiveStream associated with the chunk.
     * @param [out] consumed_packets: Reference to the number of processed packets.
     *
     * @return: Status of the operation.
     */
    ReturnStatus consume_chunk(
        const ReceiveChunk& chunk,
        const IReceiveStream& stream,
        size_t& consumed_packets) override;

private:
    // TODO: std::unique_ptr<IULPPacketBufferReader> m_buffer_reader;
    std::shared_ptr<IMediaEssenceSink> m_sink;
    std::shared_ptr<MediaUnitPool> m_media_unit_pool;
    std::shared_ptr<MediaUnit> m_current_media_unit;
};

/**
 * @brief: Configuration settings for Rivermax Dev Kit Media Receiver.
 */
struct MediaReceiverSettings : AppSettings
{
public:
    void init_default_values() override;

    /**
     * @brief: Set of enabled SMPTE standards for this receiver.
     *
     * Populated based on AppMediaSettings flags (enable_video, enable_audio, enable_ancillary).
     */
    std::set<SMPTEStandard> enabled_smpte_standards;
    /**
     * @brief: MediaSettings configurations for each enabled SMPTE standard.
     *
     * Created by MediaSettingsCalculator for each enabled standard.
     */
    std::vector<std::unique_ptr<MediaSettings>> smpte_standard_configs;
    /**
     * @brief: Mapping from media settings to the stream count of the corresponding IONode.
     *
     * Each entry represents one IONode: the first element is the media settings for that
     * IONode's media type, and the second is the number of streams assigned to it.
     */
    std::vector<std::pair<const MediaSettings&, size_t>> smpte_standard_to_nodes;
};

/**
 * @brief: Validator for Rivermax Dev Kit Media Receiver settings.
 */
class MediaReceiverSettingsValidator : public ISettingsValidator<MediaReceiverSettings>
{
public:
    /**
     * @brief: Validates media receiver settings.
     *
     * @param [in] settings: The settings to validate.
     *
     * @return: Status of the operation.
     */
    ReturnStatus validate(const MediaReceiverSettings& settings) const override;
};

/**
 * @brief: CLI settings Builder for Rivermax Dev Kit Media Receiver.
 */
class MediaReceiverCLISettingsBuilder : public CLISettingsBuilder<MediaReceiverSettings>
{
public:
    /**
     * @brief: MediaReceiverCLISettingsBuilder constructor.
     *
     * @param [in] argc: Number of CLI arguments.
     * @param [in] argv: CLI arguments strings array.
     * @param [in] app_description: Application description string for the CLI usage.
     * @param [in] app_examples: Application examples string for the CLI usage.
     * @param [in] validator: A const reference to the settings validator.
     */
    MediaReceiverCLISettingsBuilder(int argc, const char** argv,
        const std::string& app_description,
        const std::string& app_examples,
        const ISettingsValidator<MediaReceiverSettings>& validator) :
        CLISettingsBuilder<MediaReceiverSettings>(argc, argv, app_description, app_examples, validator) {}
    virtual ~MediaReceiverCLISettingsBuilder() = default;
protected:
    ReturnStatus add_cli_options(MediaReceiverSettings& settings) override;
};

using MediaReceiverUserProvidedSettingsBuilder = UserProvidedSettingsBuilder<MediaReceiverSettings>;

/**
 * @brief: Media Receiver application.
 */
class MediaReceiverApp : public ReceiverBaseApp
{
private:
    /* Settings builder pointer */
    std::unique_ptr<ISettingsBuilder<MediaReceiverSettings>> m_settings_builder;
    /* Application settings pointer */
    std::shared_ptr<MediaReceiverSettings> m_media_receiver_settings;
    /* Network receive flows */
    std::vector<ReceiveFlow> m_flows;
    /**
     * @brief: Media type configuration function map.
     *
     * This static map contains functions for configuring different SMPTE standards.
     * Each @ref SMPTEStandard enum value maps to a function that configures that specific media type.
     */
    static const std::unordered_map<SMPTEStandard, std::function<ReturnStatus(MediaReceiverApp*)>> s_smpte_standard_config_map;

public:
    /**
     * @brief: MediaReceiverApp class constructor.
     *
     * @param [in] settings_builder: Settings builder pointer.
     */
    MediaReceiverApp(std::unique_ptr<ISettingsBuilder<MediaReceiverSettings>> settings_builder);
    virtual ~MediaReceiverApp() = default;
    ReturnStatus initialize() override;
    /**
     * @brief: Initializes SMPTE standards configuration.
     *
     * This method is responsible for initializing the configuration of different SMPTE standards
     * that will be used by the receiver application.
     *
     * @return: Status of the operation.
     */
    ReturnStatus initialize_smpte_standards();
    /**
     * @brief: Sets media essence sink for a specific stream.
     *
     * This is the application-level API that forwards to the IO node level.
     * It forwards the request to the appropriate ReceiverIONode instance based on the stream index.
     * This method configures the media essence sink that receives reconstructed media units from a stream.
     *
     * @par Usage:
     * External applications implement @ref IMediaEssenceSink and register it with the receiver.
     * When complete media units (video frames, audio samples, ancillary data) are reconstructed,
     * they are delivered to the sink via @ref IMediaEssenceSink::put_media_unit_non_blocking().
     *
     * @par Default Behavior:
     * Each stream is initialized with @ref NullEssenceSink by default, which discards all
     * received media units. A real sink implementation should be set for meaningful data processing.
     *
     * @param [in] stream_index: The external stream index to configure.
     * @param [in] smpte_standard: The SMPTE standard for media formatting.
     * @param [in] essence_sink: Sink for receiving reconstructed media units.
     *                           Pass nullptr to preserve the existing sink (default: nullptr).
     *
     * @return: Status of the operation.
     */
    ReturnStatus set_media_essence_sink(
        size_t stream_index,
        SMPTEStandard smpte_standard,
        std::shared_ptr<IMediaEssenceSink> essence_sink = nullptr);

private:
    void distribute_work_for_threads() override;
    ReturnStatus initialize_app_settings() final;
    ReturnStatus post_load_settings() final;
    void configure_network_flows() final;
    void initialize_receive_io_nodes() final;
    void run_receiver_threads() final;
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
    ReturnStatus configure_media_type_helper(std::unique_ptr<SettingsType> settings,
        const std::string& smpte_standard_name);
    /**
     * @brief: Configures video types processing.
     *
     * This method is responsible for configuring video media types processing
     * for the receiver application.
     *
     * @return: Status of the operation.
     */
    ReturnStatus configure_video_types();
    /**
     * @brief: Configures audio types processing.
     *
     * This method is responsible for configuring audio media types processing
     * for the receiver application.
     *
     * @return: Status of the operation.
     */
    ReturnStatus configure_audio_types();
    /**
     * @brief: Configures ancillary types processing.
     *
     * This method is responsible for configuring ancillary media types processing
     * for the receiver application.
     *
     * @return: Status of the operation.
     */
    ReturnStatus configure_ancillary_types();
    /**
     * @brief: Configures processing of enabled SMPTE standards.
     *
     * This method is responsible for configuring processing of enabled SMPTE standards
     * for the receiver application.
     *
     * @return: Status of the operation.
     */
    ReturnStatus configure_smpte_standards_processing();
    /**
     * @brief: Sets internal media essence sinks.
     *
     * This method is responsible for setting internal (default) media essence sinks for
     * the streams. The internal media essence sinks discard received media units.
     * User will be able to set an external media essence sink by @ref MediaReceiverApp::set_media_essence_sink.
     *
     * @return: Status of the operation.
     */
    ReturnStatus set_internal_media_essence_sinks();
    /**
     * @brief: Creates a @ref MediaReconstructionDataConsumer for the given media settings and sink.
     *
     * @param [in] media_settings: Media settings for the stream.
     * @param [in] sink: Sink to deliver reconstructed media units.
     *
     * @return: Unique pointer to the new consumer.
     */
    std::unique_ptr<MediaReconstructionDataConsumer> create_consumer_for_stream(
        const MediaSettings& media_settings,
        std::shared_ptr<IMediaEssenceSink> sink);
    /**
     * @brief: Intentionally deleted to redirect users to @ref set_media_essence_sink.
     *
     * Deleted to hide the base @ref ReceiverBaseApp::set_receive_data_consumer
     * from direct callers of @ref MediaReceiverApp. Users should call
     * @ref set_media_essence_sink instead.
     */
    ReturnStatus set_receive_data_consumer(size_t stream_index,
        std::unique_ptr<IReceiveDataConsumer> data_consumer) = delete;
};

} // namespace apps
} // namespace rdk

#endif /* RDK_APPS_MEDIA_RECEIVER_MEDIA_RECEIVER_H_ */
