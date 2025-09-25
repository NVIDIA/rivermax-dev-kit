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

#include "rdk/services/ulp_packet_buffer_wr/writers/rtp_smpte_2110_30_packet_buffer_writer.h"
#include "rdk/services/media/media_defs.h"
#include "rdk/services/media/media_settings_audio.h"

using namespace rivermax::dev_kit::services;

ReturnStatus RTP_SMPTE_2110_30_MockPacketBufferWriter::set_next_media_unit(std::shared_ptr<MediaUnit> media_unit)
{
    reset_in_media_unit_state();
    return ReturnStatus::success;
}

void RTP_SMPTE_2110_30_MockPacketBufferWriter::reset_in_media_unit_state()
{
    m_rtp_packet_context->counter = 0;
}

void RTP_SMPTE_2110_30_MockPacketBufferWriter::update_in_media_unit_state(size_t header_size, size_t payload_size) 
{
    const auto& audio_settings = static_cast<const SMPTE_2110_30_MediaSettings&>(m_media_settings);
    uint32_t ticks_per_packet = static_cast<uint32_t>((m_media_settings.sample_rate * audio_settings.ptime_usec) / USEC_IN_SEC);

    m_rtp_packet_context->timestamp += ticks_per_packet;

    // Track packet counter for media unit boundaries (but doesn't affect timestamp)
    if (++m_rtp_packet_context->counter >= m_media_settings.packets_in_media_unit) {
        m_rtp_packet_context->counter = 0;
    }
    m_rtp_packet_context->sequence++;
}

ReturnStatus RTP_SMPTE_2110_30_PacketBufferWriter::set_next_media_unit(std::shared_ptr<MediaUnit> media_unit)
{
    // Todo: Implement actual media unit handling.
    return ReturnStatus::success;
}
