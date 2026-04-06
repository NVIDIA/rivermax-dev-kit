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

#ifndef RDK_SERVICES_MEDIA_ANCILLARY_ESSENCE_SOURCE_H_
#define RDK_SERVICES_MEDIA_ANCILLARY_ESSENCE_SOURCE_H_

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "rdk/services/media/afd_encoder.h"
#include "rdk/services/media/ancillary_metadata.h"
#include "rdk/services/media/closed_caption_608_encoder.h"
#include "rdk/services/media/closed_caption_source_interface.h"
#include "rdk/services/media/media_essence_source.h"
#include "rdk/services/media/media_settings_ancillary.h"
#include "rdk/services/media/timecode_encoder.h"
#include "rdk/services/utils/enum_utils.h"

namespace rdk
{
namespace services
{

/**
 * @brief: Supported ancillary data types.
 */
enum class AncillaryDataType : uint8_t
{
    None = 0,
    AFD = 1 << 0,
    Timecode = 1 << 1,
    ClosedCaption = 1 << 2,
};
RDK_ENUM_FLAGS(AncillaryDataType);

/**
 * @brief: Ancillary essence source to generate ancillary media units.
 *
 * Can generate ancillary data for the following types:
 * - AFD: Active Format Description (AFD) code and aspect ratio.
 * - Timecode: Transmission time in S12M-2 format (LTC, VITC1 and VITC2).
 * - Simple Closed Caption: CEA-608 captions encoded in CDP.
 */
class AncillaryEssenceSource : public IMediaEssenceSource
{
protected:
    const SMPTE_2110_40_MediaSettings& m_media_settings;
    AncillaryDataType m_ancillary_data_types = AncillaryDataType::None;

    uint64_t m_start_time = 0;
    uint32_t m_unit_counter = 0;

    std::unique_ptr<AFDEncoder> m_afd_encoder;
    std::unique_ptr<TimecodeEncoder> m_timecode_encoder;
    std::unique_ptr<ClosedCaption608Encoder> m_cc_encoder;
    std::unique_ptr<IClosedCaptionSource> m_cc_source;

public:
    /**
     * @brief: Constructor for AncillaryEssenceSource.
     *
     * @param [in] media_settings: Ancillary media settings for the stream.
     */
    AncillaryEssenceSource(const SMPTE_2110_40_MediaSettings& media_settings);
    std::shared_ptr<MediaUnit> get_media_unit_blocking() override;
    std::shared_ptr<MediaUnit> get_media_unit_non_blocking() override;
    void set_start_time(uint64_t time_ns) override;
    /**
     * @brief: Sets the closed caption source.
     *
     * @param [in] source: Closed caption source implementation.
     */
    void set_caption_source(std::unique_ptr<IClosedCaptionSource> source);

private:
    /**
     * @brief: Initializes ancillary data encoders based on enabled types.
     *
     * @param [in] data_identifiers: Ancillary data identifiers to enable.
     */
    void initialize_encoders(const std::vector<AncillaryDataIdentifier>& data_identifiers);
    /**
     * @brief: Processes AFD encoder.
     *
     * @param [out] buffer: Buffer to write data to.
     * @param [in,out] cumulative_offset: Current offset in the buffer.
     * @param [out] anc_metadata: Ancillary metadata to update.
     */
    void process_afd_encoder(uint8_t* buffer, size_t& cumulative_offset, AncillaryMediaUnitMetadata& anc_metadata);
    /**
     * @brief: Processes timecode encoder.
     *
     * @param [out] buffer: Buffer to write data to.
     * @param [in,out] cumulative_offset: Current offset in the buffer.
     * @param [out] anc_metadata: Ancillary metadata to update.
     */
    void process_timecode_encoder(uint8_t* buffer, size_t& cumulative_offset, AncillaryMediaUnitMetadata& anc_metadata);
    /**
     * @brief: Processes closed caption encoder.
     *
     * @param [out] buffer: Buffer to write data to.
     * @param [in,out] cumulative_offset: Current offset in the buffer.
     * @param [out] anc_metadata: Ancillary metadata to update.
     */
    void process_closed_caption_encoder(uint8_t* buffer, size_t& cumulative_offset, AncillaryMediaUnitMetadata& anc_metadata);
    /**
     * @brief: Writes encoder data to buffer and updates metadata.
     *
     * @param [in] encoder: Ancillary data encoder.
     * @param [out] buffer: Buffer to write data to.
     * @param [in,out] cumulative_offset: Current offset in the buffer.
     * @param [out] anc_metadata: Ancillary metadata to update.
     */
    void write_encoder_data(IAncillaryDataEncoder& encoder, uint8_t* buffer, size_t& cumulative_offset, AncillaryMediaUnitMetadata& anc_metadata);
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_MEDIA_ANCILLARY_ESSENCE_SOURCE_H_ */
