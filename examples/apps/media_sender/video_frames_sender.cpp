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

#include "rdk/apps/media_sender/media_sender.h"
#include "rdk/services/media/media_defs.h"
#include "rdk/services/media/media_settings_video.h"
#include "rdk/services/media/null_essence_source.h"
#include "rdk/examples/apps/media_sender/video_frames_sender.h"

using namespace rdk::apps;
using namespace rdk::services;
using namespace rdk::examples::apps;

namespace rdk
{
namespace examples
{
namespace apps
{

constexpr const char* EXAMPLE_DESCRIPTION = "NVIDIA Rivermax Dev Kit Video Frames Sender Example";
constexpr const char* EXAMPLE_USAGE = "Examples:\n"
                                      "  1. video_frames_sender --local-ip 1.2.3.4";

VideoFramesSenderExample::VideoFramesSenderExample() :
    BaseExample(EXAMPLE_DESCRIPTION, EXAMPLE_USAGE)
{
}

void VideoFramesSenderExample::configure_media_sender_settings(MediaSenderSettings& settings) const
{
    settings.init_default_values();

    /** Configure network settings */
    settings.local_ips = {m_settings->local_ip};
    settings.destination_ips = {"224.1.1.1"};
    settings.destination_ports = {2000};

    /** Configure video settings */
    settings.media.enable_video = true;
    settings.media.resolution = {1920, 1080};
    settings.media.frame_rate = 60;
    settings.media.sampling_type = VideoSampling::YCbCr_4_2_2;
    settings.media.color_bit_depth = VideoBitDepth::_10;

    /** Configure thread/stream settings */
    settings.num_of_threads = 1;
    settings.num_of_total_streams = 1;
    settings.internal_thread_core = 0;
    settings.app_threads_cores = {1};

    /** Configure memory settings */
    settings.allocator_type = AllocatorTypeUI::Auto;
    settings.app_memory_alloc = true;
    settings.header_data_split = true;
}

ReturnStatus VideoFramesSenderExample::operator()()
{
    /** 1. Configure media sender settings with user parameters */
    MediaSenderSettings media_sender_settings;

    media_sender_settings.init_default_values();
    configure_media_sender_settings(media_sender_settings);

    /**
     * 2. Initialize a custom media essence source.
     *
     * Create a dummy media essence source using @ref NullEssenceSource with 0.2f probability of simulating
     * media unit (i.e., video frame) unavailability.
     * This source implements the @ref IMediaEssenceSource interface, which client
     * application should implement to supply media units (i.e., video frames) to the Media Sender application.
     */
    SMPTE_2110_20_MediaSettings st_2110_20_settings(media_sender_settings);
    auto dummy_media_essence_source = std::make_shared<NullEssenceSource>(st_2110_20_settings);
    constexpr float probability = 0.2f;
    auto status = dummy_media_essence_source->set_media_unit_not_available_probability(probability);
    RETURN_FAILURE_ON_ERROR(status, "Failed to set media unit unavailability probability");

    /** 3. Set up media sender settings validation */
    MediaSenderSettingsValidator settings_validator;

    /** 4. Configure user-provided settings builder with validator */
    auto settings_builder = std::make_unique<MediaSenderUserProvidedSettingsBuilder>(
        std::move(media_sender_settings), settings_validator);

    /** 5. Initialize the media sender application */
    MediaSenderApp app(std::move(settings_builder));
    status = app.initialize();
    RETURN_FAILURE_ON_ERROR(status, "Failed to initialize Media Sender application");

    /**
     * 6. Set the custom media essence source to the Media Sender application.
     *
     * In this example, we use a runtime-only media essence source configuration:
     * - @p preload_essence_source = nullptr: No preloading of data into memory blocks, before transmission starts.
     * - @p runtime_essence_source = @p dummy_media_essence_source: Provides media units dynamically during transmission.
     * - @p runtime_contains_payload = false: Only RTP headers are generated, no payload data is copied, during transmission.
     *   This improves performance when payload data is not needed or handled before transmission.
     */
    constexpr size_t stream_index = 0;
    constexpr SMPTEStandard smpte_standard = SMPTEStandard::ST_2110_20;
    std::shared_ptr<IMediaEssenceSource> preload_essence_source = nullptr;
    std::shared_ptr<IMediaEssenceSource> runtime_essence_source = dummy_media_essence_source;
    constexpr bool runtime_contains_payload = false;
    status = app.set_media_essence_sources(
        stream_index, smpte_standard, std::move(preload_essence_source),
        std::move(runtime_essence_source), runtime_contains_payload);
    RETURN_FAILURE_ON_ERROR(status, "Failed to set media essence source");

    /** 7. Start sending media data */
    status = app.run();
    RETURN_FAILURE_ON_ERROR(status, "Failed to run Media Sender application");

    return ReturnStatus::success;
}

} // namespace apps
} // namespace examples
} // namespace rdk

int main(int argc, const char* argv[])
{
    return rdk::examples::common_example_main<VideoFramesSenderExample>(argc, argv);
}
