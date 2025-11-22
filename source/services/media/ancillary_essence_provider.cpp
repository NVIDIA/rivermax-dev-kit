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

#include <algorithm>
#include <iostream>

#include "rdk/services/media/ancillary_essence_provider.h"
#include "rdk/services/media/closed_caption_mock_source.h"

using namespace rivermax::dev_kit::services;

AncillaryEssenceProvider::AncillaryEssenceProvider(const SMPTE_2110_40_MediaSettings& media_settings) :
    m_media_settings(media_settings)
{
    initialize_encoders(media_settings.data_identifiers);
}

std::shared_ptr<MediaUnit> AncillaryEssenceProvider::get_media_unit_blocking()
{
    // Ancillary data, AFD, timecode, and closed captions are generated on-the-fly
    // and never block. Valid data is always available.
    return get_media_unit_non_blocking();
}

std::shared_ptr<MediaUnit> AncillaryEssenceProvider::get_media_unit_non_blocking()
{
    auto media_unit = std::make_shared<MediaUnit>(m_media_settings.bytes_per_media_unit, SMPTEStandard::ST_2110_40);
    uint8_t* buffer = reinterpret_cast<uint8_t*>(media_unit->data->get());
    size_t cumulative_offset = 0;
    auto anc_metadata = std::static_pointer_cast<AncillaryMediaUnitMetadata>(media_unit->metadata);

    process_afd_encoder(buffer, cumulative_offset, *anc_metadata);
    process_timecode_encoder(buffer, cumulative_offset, *anc_metadata);
    process_closed_caption_encoder(buffer, cumulative_offset, *anc_metadata);

    m_unit_counter++;

    return media_unit;
}

void AncillaryEssenceProvider::set_start_time(uint64_t time_ns)
{
    m_start_time = time_ns;
    m_unit_counter = 0;

    if (m_timecode_encoder) {
        m_timecode_encoder->set_start_time(time_ns);
    }
}

void AncillaryEssenceProvider::set_caption_source(std::unique_ptr<IClosedCaptionSource> source)
{
    m_cc_source = std::move(source);
}

void AncillaryEssenceProvider::initialize_encoders(const std::vector<AncillaryDataIdentifier>& data_identifiers)
{
    uint32_t fps = m_media_settings.frame_rate.num / m_media_settings.frame_rate.denom;

    for (const auto& identifier : data_identifiers) {
        if (identifier == ANCILLARY_TIMECODE_IDENTIFIER) {
            std::vector<TimecodePayloadType> payload_types = { TimecodePayloadType::LTC, TimecodePayloadType::VITC1 };
            m_timecode_encoder = std::make_unique<TimecodeEncoder>(fps, m_start_time, payload_types);
        } else if (identifier == ANCILLARY_AFD_IDENTIFIER) {
            m_afd_encoder = std::make_unique<AFDEncoder>(AFDCode::FullFrame, AFDAspectRatio::Aspect_16x9);
        } else if (identifier == ANCILLARY_CLOSED_CAPTION_IDENTIFIER) {
            m_cc_encoder = std::make_unique<ClosedCaption608Encoder>(fps);
            m_cc_source = std::make_unique<ClosedCaptionMockSource>();
        } else {
            std::cout << "Unsupported ancillary data identifier: " << identifier.did << ":" << identifier.sdid
                      << std::endl;
        }
    }
}

void AncillaryEssenceProvider::process_afd_encoder(uint8_t* buffer, size_t& cumulative_offset, AncillaryMediaUnitMetadata& anc_metadata)
{
    if (m_afd_encoder) {
        write_encoder_data(*m_afd_encoder, buffer, cumulative_offset, anc_metadata);
    }
}

void AncillaryEssenceProvider::process_timecode_encoder(uint8_t* buffer, size_t& cumulative_offset, AncillaryMediaUnitMetadata& anc_metadata)
{
    if (m_timecode_encoder) {
        m_timecode_encoder->set_current_frame(m_unit_counter);
        write_encoder_data(*m_timecode_encoder, buffer, cumulative_offset, anc_metadata);
    }
}

void AncillaryEssenceProvider::process_closed_caption_encoder(uint8_t* buffer, size_t& cumulative_offset, AncillaryMediaUnitMetadata& anc_metadata)
{
    if (m_cc_encoder && m_cc_source) {
        uint64_t relative_timestamp_ns = m_unit_counter * m_media_settings.media_unit_time_interval_ns;
        std::string current_text = m_cc_source->get_caption_text(relative_timestamp_ns);

        m_cc_encoder->update(current_text);
        write_encoder_data(*m_cc_encoder, buffer, cumulative_offset, anc_metadata);
    }
}

void AncillaryEssenceProvider::write_encoder_data(IAncillaryDataEncoder& encoder, uint8_t* buffer, size_t& cumulative_offset, AncillaryMediaUnitMetadata& anc_metadata)
{
    size_t packet_count = encoder.get_packet_count();
    for (size_t i = 0; i < packet_count; ++i) {
        AncillaryDataDescriptor desc{};
        size_t data_size = encoder.write_data(i, &buffer[cumulative_offset]);
        encoder.fill_descriptor_header(i, desc.ancillary_data_header);
        desc.ancillary_data_header.user_data_words_count = static_cast<uint16_t>(data_size);
        desc.user_data_offset = cumulative_offset;

        cumulative_offset += data_size;
        anc_metadata.ancillary_data.push_back(desc);
    }
}
