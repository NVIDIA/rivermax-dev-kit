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

#include <chrono>
#include <cstdint>
#include <memory>

#include "rdk/services/utils/defs.h"
#include "rdk/services/settings/app_settings.h"
#include "rdk/services/ulp_packet_buffer/writers/rtp_smpte_2110_20_packet_buffer_writer.h"
#include "rdk/services/media/media_settings_video.h"
#include "rdk/examples/services/ulp_packet_buffer/writers/rtp_smpte_2110_20_packet_buffer_writer.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::examples::services;

namespace rivermax
{
namespace dev_kit
{
namespace examples
{
namespace services
{

constexpr const char* EXAMPLE_DESCRIPTION = "NVIDIA Rivermax Dev Kit SMPTE 2110-20 RTP Packet Buffer Writer Example";
constexpr const char* EXAMPLE_USAGE = "Examples:\n"
                                      "  1. rtp_smpte_2110_20_packet_buffer_writer";

RTP_SMPTE_2110_20_PacketBufferWriterExample::RTP_SMPTE_2110_20_PacketBufferWriterExample() :
    BaseExample(EXAMPLE_DESCRIPTION, EXAMPLE_USAGE)
{
}

void RTP_SMPTE_2110_20_PacketBufferWriterExample::configure_app_settings(AppSettings& settings) const
{
    /** Configure network settings */
    settings.local_ip = "1.1.1.1";

    /** Configure video settings */
    settings.media.enable_video = true;
    settings.media.resolution = {1920, 1080};
    settings.media.frame_rate = 60;
    settings.media.sampling_type = VideoSampling::YCbCr_4_2_2;
    settings.media.color_bit_depth = VideoBitDepth::_10;

    /** Configure memory settings */
    settings.header_data_split = true;
}

ReturnStatus RTP_SMPTE_2110_20_PacketBufferWriterExample::operator()()
{
    /** 1. Configure application settings with user parameters */
    auto app_settings = std::make_shared<AppSettings>();

    app_settings->init_default_values();
    configure_app_settings(*app_settings);

    /** 2. Create and configure media settings */
    auto st_2110_20_settings = std::make_unique<SMPTE_2110_20_MediaSettings>(*app_settings);
    auto status = st_2110_20_settings->create_default_calculator();
    RETURN_FAILURE_ON_ERROR(status, "Failed to create media settings calculator");

    /** 3. Create RTP media packet buffer writer */
    constexpr SMPTEStandard smpte_standard = SMPTEStandard::ST_2110_20;
    constexpr bool contains_payload = true;
    auto memory_allocator = m_facade.get_memory_allocator(AllocatorType::HugePageDefault, std::move(app_settings));
    RETURN_STATUS_IF(!memory_allocator, "Failed to get memory allocator", ReturnStatus::failure);
    auto header_memory_utils = memory_allocator->get_memory_utils();
    auto payload_memory_utils = memory_allocator->get_memory_utils();

    auto packet_buffer_writer = rivermax::dev_kit::services::create_rtp_media_packet_buffer_writer(
        smpte_standard,
        contains_payload,
        *st_2110_20_settings,
        std::move(header_memory_utils),
        std::move(payload_memory_utils));
    RETURN_STATUS_IF(!packet_buffer_writer, "Failed to create RTP media packet buffer writer", ReturnStatus::failure);

    /** 4. Set the start time for the first packet */
    uint64_t send_time_ns = std::chrono::steady_clock::now().time_since_epoch().count() + NS_IN_SEC;
    uint64_t aligned_time_ns = static_cast<uint64_t>(
        st_2110_20_settings->media_settings_calculator->align_time_to_media_unit_boundary_ns(send_time_ns));
    packet_buffer_writer->set_start_time(aligned_time_ns);

    /**
     * 5. Prepare the media unit.
     *
     * Note: In the context of SMPTE ST 2110-20 video, a media unit represents a single video frame.
     */

    /** 5.1 Create a media unit object */
    size_t media_unit_size = st_2110_20_settings->bytes_per_media_unit;
    auto media_unit = std::make_shared<MediaUnit>(media_unit_size, SMPTEStandard::ST_2110_20);

    /** 5.2 Generate the media unit data, here we just fill it with zeros for demonstration */
    auto media_unit_memory_utils = memory_allocator->get_memory_utils();
    status = media_unit_memory_utils->memory_set(media_unit->data->get(), 0, media_unit_size);
    RETURN_FAILURE_ON_ERROR(status, "Failed to write media unit data");

    /** 5.3 Set the next media unit for writing to the packet IO buffer */
    status = packet_buffer_writer->set_next_media_unit(std::move(media_unit));
    RETURN_FAILURE_ON_ERROR(status, "Failed to set next media unit");

    /** 6. Allocate memory for the header and payload */
    uint32_t packets_in_media_unit = st_2110_20_settings->packets_in_media_unit;
    auto header_memory_size = st_2110_20_settings->app_header_stride_size * packets_in_media_unit;
    auto payload_memory_size = st_2110_20_settings->data_stride_size * packets_in_media_unit;
    auto header_memory_ptr = memory_allocator->allocate(header_memory_size);
    auto payload_memory_ptr = memory_allocator->allocate(payload_memory_size);
    RETURN_STATUS_IF(!header_memory_ptr || !payload_memory_ptr, "Failed to allocate memory", ReturnStatus::failure);

    /**
     * 7. Write packets to the packet IO buffer.
     *
     * Here we write the entire media unit at once, but in a real application,
     * this could be done in smaller chunks or in a loop for continuous streaming.
     */
    status = packet_buffer_writer->write_buffer(header_memory_ptr, payload_memory_ptr, packets_in_media_unit);
    RETURN_FAILURE_ON_ERROR(status, "Failed to write buffer");

    return ReturnStatus::success;
}

} // namespace services
} // namespace examples
} // namespace dev_kit
} // namespace rivermax

int main(int argc, const char* argv[])
{
    return rivermax::dev_kit::examples::common_example_main<RTP_SMPTE_2110_20_PacketBufferWriterExample>(argc, argv);
}
