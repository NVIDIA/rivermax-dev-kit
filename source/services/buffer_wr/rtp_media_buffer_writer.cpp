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

#include "rt_threads.h"

#include "rdk/services/buffer_wr/rtp_ancillary_buffer_writer.h"
#include "rdk/services/buffer_wr/rtp_audio_buffer_writer.h"
#include "rdk/services/buffer_wr/rtp_media_buffer_writer.h"
#include "rdk/services/buffer_wr/rtp_video_buffer_writer.h"

using namespace rivermax::dev_kit::services;

struct RTPHeader {
    uint8_t cc : 4;            // CSRC count
    uint8_t extension : 1;     // Extension bit
    uint8_t padding : 1;       // Padding bit
    uint8_t version : 2;       // Version, currently 2
    uint8_t payload_type : 7;  // Payload type
    uint8_t marker : 1;        // Marker bit
    uint16_t sequence_number;  // Sequence number
    uint32_t timestamp;        // Timestamp
    uint32_t ssrc;             // Synchronization source (SSRC) identifier
};

RTPMediaBufferWriter::RTPMediaBufferWriter(const MediaSettings& media_settings,
    std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils) :
    IBufferWriter(std::move(header_mem_utils), std::move(payload_mem_utils)),
    m_media_settings(media_settings),
    m_ssrc(DEFAULT_SSRC) // Simulated SSRC.
{
    set_stream_properties();
}

ReturnStatus RTPMediaBufferWriter::write_buffer(void* payload_ptr, size_t length_in_strides)
{
    byte_t* header_pointer = reinterpret_cast<byte_t*>(payload_ptr);
    assert(header_pointer);
    uint64_t stride = 0;
    byte_t* current_packet_pointer;
    byte_t* current_payload_pointer;

    while (stride < length_in_strides && m_send_data.packet_counter < m_media_settings.packets_in_media_unit) {
        current_packet_pointer = header_pointer + (stride * m_media_settings.data_stride_size);
        current_payload_pointer = (current_packet_pointer + m_media_settings.protocol_header_size);
        build_rtp_header(current_packet_pointer);
        fill_packet(current_payload_pointer);
        update_in_media_unit_state();
        stride++;
    }
    return ReturnStatus::success;
}

ReturnStatus RTPMediaBufferWriter::write_buffer(void* header_ptr, void* payload_ptr, size_t length_in_strides)
{
    assert(payload_ptr);
    assert(header_ptr);

    byte_t* header_pointer = reinterpret_cast<byte_t*>(header_ptr);
    byte_t* payload_pointer = reinterpret_cast<byte_t*>(payload_ptr);
    assert(header_pointer);
    assert(payload_ptr);
    uint64_t stride = 0;
    byte_t* current_packet_pointer;
    byte_t* current_payload_pointer;

    while (stride < length_in_strides && m_send_data.packet_counter < m_media_settings.packets_in_media_unit) {
        current_packet_pointer = header_pointer + (stride * m_media_settings.app_header_stride_size);
        current_payload_pointer = (payload_pointer + (stride * m_media_settings.data_stride_size));
        build_rtp_header(current_packet_pointer);
        fill_packet(current_payload_pointer);
        update_in_media_unit_state();
        stride++;
    }
    return ReturnStatus::success;
}

size_t RTPMediaBufferWriter::build_rtp_header_common(byte_t* buffer)
{
    // build RTP header - 12 bytes:
    /*
     0                   1                   2                   3
     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     | V |P|X|  CC   |M|     PT      |            SEQ                |
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     |                           timestamp                           |
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     |                           ssrc                                |
     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+*/

    RTPHeader* p_rtp_header = reinterpret_cast<RTPHeader*>(buffer);
    p_rtp_header->version = 2;
    p_rtp_header->padding = 0;
    p_rtp_header->extension = 0;
    p_rtp_header->cc = 0;
    p_rtp_header->payload_type = m_media_settings.payload_type;
    p_rtp_header->sequence_number = htons(static_cast<uint16_t>(m_send_data.rtp_sequence));
    p_rtp_header->timestamp = htonl(static_cast<uint32_t>(m_send_data.rtp_timestamp));
    p_rtp_header->ssrc = htonl(m_ssrc);
    p_rtp_header->marker = (m_send_data.packet_counter == m_media_settings.packets_in_media_unit - 1) ? 1 : 0;

    return sizeof(RTPHeader);
}

void RTPMediaBufferWriter::set_first_packet_timestamp(uint64_t packet_time_ns)
{
    m_send_data.rtp_timestamp = static_cast<uint32_t>(
        time_to_rtp_timestamp(packet_time_ns, static_cast<int>(m_media_settings.sample_rate)));
}

rtp_media_buffer_writer_factory_map_t RTPMediaBufferWriter::s_rtp_media_buffer_writer_factory = \
{
    {
        {SMPTEStandard::ST_2110_20, false},
        [](const MediaSettings& media_settings,
            std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils)
        {
            return std::unique_ptr<RTPMediaBufferWriter>(new RTPVideoMockBufferWriter(media_settings,
                std::move(header_mem_utils), std::move(payload_mem_utils)));
        }
    },
    {
        {SMPTEStandard::ST_2110_20, true},
        [](const MediaSettings& media_settings,
            std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils)
        {
            return std::unique_ptr<RTPMediaBufferWriter>(new RTPVideoBufferWriter(media_settings,
                std::move(header_mem_utils), std::move(payload_mem_utils)));
        }
    },
    {
        {SMPTEStandard::ST_2110_30, false},
        [](const MediaSettings& media_settings,
            std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils)
        {
            return std::unique_ptr<RTPMediaBufferWriter>(new RTPAudioBufferWriter(media_settings,
                std::move(header_mem_utils), std::move(payload_mem_utils)));
        }
    },
    {
        {SMPTEStandard::ST_2110_30, true},
        [](const MediaSettings& media_settings,
            std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils)
        {
            return std::unique_ptr<RTPMediaBufferWriter>(new RTPAudioBufferWriter(media_settings,
                std::move(header_mem_utils), std::move(payload_mem_utils)));
        }
    },
    {
        {SMPTEStandard::ST_2110_40, false},
        [](const MediaSettings& media_settings,
            std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils)
        {
            return std::unique_ptr<RTPMediaBufferWriter>(new RTPAncillaryBufferWriter(media_settings,
                std::move(header_mem_utils), std::move(payload_mem_utils)));
        }
    },
    {
        {SMPTEStandard::ST_2110_40, true},
        [](const MediaSettings& media_settings,
            std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils)
        {
            return std::unique_ptr<RTPMediaBufferWriter>(new RTPAncillaryBufferWriter(media_settings,
                std::move(header_mem_utils), std::move(payload_mem_utils)));
        }
    },
};

std::unique_ptr<RTPMediaBufferWriter> RTPMediaBufferWriter::get_rtp_media_buffer_writer(
    SMPTEStandard type, bool contains_payload, const MediaSettings& media_settings,
    std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils)
{
    auto key = MediaBufferFactoryKey(type, contains_payload);
    auto iter = RTPMediaBufferWriter::s_rtp_media_buffer_writer_factory.find(key);
    if (iter != RTPMediaBufferWriter::s_rtp_media_buffer_writer_factory.end()) {
        return iter->second(media_settings, std::move(header_mem_utils), std::move(payload_mem_utils));
    } else {
        return nullptr;
    }
}
