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
#include <memory>

#include "rdk/services/ulp_packet_buffer/writers/rtp_smpte_2110_20_packet_buffer_writer.h"
#include "rdk/services/media/media_unit_pool.h"
#include "rdk/services/media/media_settings_video.h"

using namespace rivermax::dev_kit::services;

RTP_SMPTE_2110_20_PacketBufferWriter::RTP_SMPTE_2110_20_PacketBufferWriter(const MediaSettings& media_settings,
    std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils, bool enable_zero_copy)
    : RTPMediaPacketBufferWriter<RTP_SMPTE_2110_20_PacketContext, RTP_SMPTE_2110_20_PacketWriter>(
        media_settings, std::move(header_mem_utils), std::move(payload_mem_utils), enable_zero_copy)
{
    m_rtp_packet_context->srd_length = media_settings.raw_packet_payload_size;
}

void RTP_SMPTE_2110_20_PacketBufferWriter::reset_in_media_unit_state()
{
    m_current_line_number = 0;
    m_current_srd_offset = 0;
}

void RTP_SMPTE_2110_20_PacketBufferWriter::prepare_context_for_packet()
{
    auto& video_settings = static_cast<const SMPTE_2110_20_MediaSettings&>(m_media_settings);

    // Set source data pointer for payload (nullptr in zero copy mode)
    if (!m_zero_copy_enabled && m_current_media_unit) {
        m_rtp_packet_context->payload_ptr = m_current_media_unit->data->get() + m_bytes_consumed;
    } else {
        m_rtp_packet_context->payload_ptr = nullptr;
    }

    // Copy tracking state to context for packet writer
    m_rtp_packet_context->line_number = m_current_line_number;
    m_rtp_packet_context->srd_offset = m_current_srd_offset;
    m_rtp_packet_context->rtp_interlace_field_indicator = m_current_field_indicator;
    m_rtp_packet_context->srd_length = video_settings.raw_packet_payload_size;

    // Set marker bit based on whether this is the last packet
    m_rtp_packet_context->marker = (m_packet_counter == video_settings.packets_in_media_unit - 1) ? 1 : 0;
}

inline void RTP_SMPTE_2110_20_PacketBufferWriter::update_in_media_unit_state(size_t header_size, size_t payload_size)
{
    auto& video_settings = static_cast<const SMPTE_2110_20_MediaSettings&>(m_media_settings);

    // Update tracking state for next packet
    m_current_srd_offset = (m_current_srd_offset + video_settings.pixels_per_packet) %
        (video_settings.resolution.width);

    m_packet_counter++;

    if (!(m_packet_counter % video_settings.packets_in_line)) {
        // Prepare line number for next iteration:
        m_current_line_number = (m_current_line_number + 1) % video_settings.lines_in_frame_field;
    }

    if (m_packet_counter == video_settings.packets_in_media_unit) {
        // ST2110-20: the timestamp SHOULD be the same for each packet of the frame/field.
        m_rtp_packet_context->timestamp += video_settings.ticks_per_media_unit;
        m_packet_counter = 0;
        if (video_settings.video_scan_type == VideoScanType::Interlaced) {
            m_current_field_indicator = !m_current_field_indicator;
        }
    }

    m_rtp_packet_context->sequence++;
    m_rtp_packet_context->extended_sequence_number++;

    m_bytes_consumed += payload_size;
    if (m_current_media_unit && m_bytes_consumed >= m_current_media_unit->data->get_size()) {
        m_current_media_unit = nullptr;
    }
}
