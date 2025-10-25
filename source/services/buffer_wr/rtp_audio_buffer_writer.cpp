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

#include <cassert>
#include <cstddef>
#include <cstring>

#include "rdk/services/buffer_wr/rtp_audio_buffer_writer.h"
#include "rdk/services/media/media_defs.h"
#include "rdk/services/media/media_settings_audio.h"

using namespace rivermax::dev_kit::services;

RTPAudioBufferWriter::RTPAudioBufferWriter(const MediaSettings& media_settings,
    std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils) :
    RTPMediaBufferWriter(media_settings, std::move(header_mem_utils), std::move(payload_mem_utils))
{
    set_stream_properties();
}

ReturnStatus RTPAudioBufferWriter::set_next_media_unit(std::shared_ptr<MediaUnit> media_unit)
{
    reset_in_media_unit_state();
    return ReturnStatus::success;
}

void RTPAudioBufferWriter::reset_in_media_unit_state()
{
    m_send_data.packet_counter = 0;
}

void RTPAudioBufferWriter::update_in_media_unit_state()
{
    const auto& audio_settings = static_cast<const SMPTE_2110_30_MediaSettings&>(m_media_settings);
    uint32_t ticks_per_packet = static_cast<uint32_t>((m_media_settings.sample_rate * audio_settings.ptime_usec) / USEC_IN_SEC);

    m_send_data.rtp_timestamp += ticks_per_packet;

    // Track packet counter for media unit boundaries (but doesn't affect timestamp)
    if (++m_send_data.packet_counter >= m_media_settings.packets_in_media_unit) {
        m_send_data.packet_counter = 0;
    }
    m_send_data.rtp_sequence++;
}

size_t RTPAudioBufferWriter::build_rtp_header(byte_t* buffer)
{
    // ST 2110-30: Standard RTP header only, no extension
    // Note: Marker bit is set in build_rtp_header_common: Set to 0 for all audio packets)
    size_t rtp_header_size = build_rtp_header_common(buffer);
    buffer[1] &= 0x7F;  // Clear marker bit (bit 7 of byte 1)

    return rtp_header_size;
}

size_t RTPAudioBufferWriter::fill_packet(byte_t* buffer)
{
    // Mock implementation: no actual payload filling
    // Real implementation would copy audio samples here
    return m_media_settings.raw_packet_payload_size;
}
