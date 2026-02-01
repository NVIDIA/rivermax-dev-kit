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

#include <cassert>
#include <cstddef>
#include <cstring>

#include "rdk/services/ulp_packet_buffer/writers/rtp_smpte_2110_30_packet_buffer_writer.h"
#include "rdk/services/media/media_defs.h"
#include "rdk/services/media/media_settings_audio.h"

using namespace rivermax::dev_kit::services;

RTP_SMPTE_2110_30_PacketBufferWriter::RTP_SMPTE_2110_30_PacketBufferWriter(const MediaSettings& media_settings,
    std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils, bool enable_mock_mode)
    : RTPMediaPacketBufferWriter<RTPPacketContext, RTP_SMPTE_2110_30_PacketWriter>(
        media_settings, std::move(header_mem_utils), std::move(payload_mem_utils), enable_mock_mode)
{
    if (enable_mock_mode) {
        m_rtp_packet_writer = std::make_unique<RTP_SMPTE_2110_30_MockPacketWriter>(nullptr, nullptr);
    }
}

void RTP_SMPTE_2110_30_PacketBufferWriter::reset_in_media_unit_state()
{
    m_rtp_packet_context->counter = 0;
}

void RTP_SMPTE_2110_30_PacketBufferWriter::update_in_media_unit_state(size_t header_size, size_t payload_size)
{
    const auto& audio_settings = static_cast<const SMPTE_2110_30_MediaSettings&>(m_media_settings);
    Rational ticks_per_packet(m_media_settings.sample_rate * audio_settings.ptime_usec, USEC_IN_SEC);

    m_rtp_packet_context->timestamp += ticks_per_packet;

    // Track packet counter for media unit boundaries (but doesn't affect timestamp)
    if (++m_rtp_packet_context->counter >= m_media_settings.packets_in_media_unit) {
        m_rtp_packet_context->counter = 0;
    }
    m_rtp_packet_context->sequence++;

    m_rtp_packet_context->data_left_in_media_unit_in_bytes -= payload_size;
    if (m_rtp_packet_context->data_left_in_media_unit_in_bytes == 0) {
        m_rtp_packet_context->current_media_unit = nullptr;
    }
}
