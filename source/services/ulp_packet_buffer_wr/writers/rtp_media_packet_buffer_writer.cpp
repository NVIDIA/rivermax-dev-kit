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

#include "rdk/services/ulp_packet_buffer_wr/writers/rtp_media_packet_buffer_writer.h"
#include "rdk/services/ulp_packet_buffer_wr/writers/rtp_smpte_2110_20_packet_buffer_writer.h"
#include "rdk/services/ulp_packet_buffer_wr/writers/rtp_smpte_2110_30_packet_buffer_writer.h"
#include "rdk/services/ulp_packet_buffer_wr/writers/rtp_smpte_2110_40_packet_buffer_writer.h"

using namespace rivermax::dev_kit::services;

template<typename PacketContextType>
RTPMediaPacketBufferWriter<PacketContextType>::RTPMediaPacketBufferWriter(const MediaSettings& media_settings,
    std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils) :
    IULPPacketBufferWriter(std::move(header_mem_utils), std::move(payload_mem_utils)),
    m_media_settings(media_settings)
{
    m_rtp_packet_context = std::make_unique<PacketContextType>();
    m_rtp_packet_context->ssrc = DEFAULT_SSRC; // Simulated SSRC.
    m_rtp_packet_context->payload_type = media_settings.payload_type;
    m_rtp_packet_context->payload_size = media_settings.raw_packet_payload_size;
}

template<typename PacketContextType>
ReturnStatus RTPMediaPacketBufferWriter<PacketContextType>::write_buffer(void* payload_ptr, size_t length_in_strides)
{
    byte_t* current_packet_pointer = reinterpret_cast<byte_t*>(payload_ptr);
    assert(current_packet_pointer);
    uint64_t stride = 0;
    size_t header_size = 0;
    size_t payload_size = 0;
    ReturnStatus status = ReturnStatus::success;
    
    while (stride < length_in_strides && m_rtp_packet_context->counter < m_media_settings.packets_in_media_unit) {
        // No Header Data split mode
        auto packet = create_packet(current_packet_pointer);
        status = packet->fill_header(*m_rtp_packet_context, header_size, m_header_mem_utils.get());
        status = packet->fill_payload(*m_rtp_packet_context, payload_size, m_payload_mem_utils.get());
        update_in_media_unit_state(header_size, payload_size);
        current_packet_pointer += m_media_settings.data_stride_size;
        stride++;
    }
    return status;
}

template<typename PacketContextType>
ReturnStatus RTPMediaPacketBufferWriter<PacketContextType>::write_buffer(void* header_ptr, void* payload_ptr, size_t length_in_strides)
{
    byte_t* current_header_pointer = reinterpret_cast<byte_t*>(header_ptr);
    byte_t* current_payload_pointer = reinterpret_cast<byte_t*>(payload_ptr);
    assert(current_header_pointer);
    assert(current_payload_pointer);
    uint64_t stride = 0;
    size_t header_size = 0;
    size_t payload_size = 0;
    ReturnStatus status = ReturnStatus::success;

    while (stride < length_in_strides && m_rtp_packet_context->counter < m_media_settings.packets_in_media_unit) {
        auto packet = create_packet(current_header_pointer, current_payload_pointer); // Header Data Split mode
        status = packet->fill_header(*m_rtp_packet_context, header_size, m_header_mem_utils.get());
        status = packet->fill_payload(*m_rtp_packet_context, payload_size, m_payload_mem_utils.get());
        update_in_media_unit_state(header_size, payload_size);
        current_header_pointer += m_media_settings.app_header_stride_size;
        current_payload_pointer += m_media_settings.data_stride_size;
        stride++;
    }
    return ReturnStatus::success;
}

template<typename PacketContextType>
void RTPMediaPacketBufferWriter<PacketContextType>::set_initial_timestamp(uint64_t packet_time_ns)
{
    m_rtp_packet_context->timestamp = static_cast<uint32_t>(
        time_to_rtp_timestamp(packet_time_ns, static_cast<int>(m_media_settings.sample_rate)));
}


template<typename WriterType>
std::unique_ptr<IULPPacketBufferWriter> create_writer(
    const MediaSettings& media_settings,
    std::shared_ptr<MemoryUtils> header_mem_utils,
    std::shared_ptr<MemoryUtils> payload_mem_utils)
{
    return std::unique_ptr<WriterType>(new WriterType(media_settings,
        std::move(header_mem_utils), std::move(payload_mem_utils)));
}

static rtp_media_packet_buffer_writer_factory_map_t s_rtp_media_packet_buffer_writer_factory = {
    {{SMPTEStandard::ST_2110_20, false}, create_writer<RTP_SMPTE_2110_20_MockPacketBufferWriter>},
    {{SMPTEStandard::ST_2110_20, true},  create_writer<RTP_SMPTE_2110_20_PacketBufferWriter>},
    {{SMPTEStandard::ST_2110_30, false}, create_writer<RTP_SMPTE_2110_30_MockPacketBufferWriter>},
    {{SMPTEStandard::ST_2110_30, true},  create_writer<RTP_SMPTE_2110_30_PacketBufferWriter>},
    {{SMPTEStandard::ST_2110_40, false}, create_writer<RTP_SMPTE_2110_40_MockPacketBufferWriter>},
    {{SMPTEStandard::ST_2110_40, true},  create_writer<RTP_SMPTE_2110_40_PacketBufferWriter>}
};

std::unique_ptr<IULPPacketBufferWriter> factory::create_rtp_media_packet_buffer_writer(
    SMPTEStandard type, bool contains_payload, const MediaSettings& media_settings,
    std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils)
{
    auto key = MediaBufferFactoryKey(type, contains_payload);
    auto iter = s_rtp_media_packet_buffer_writer_factory.find(key);
    if (iter != s_rtp_media_packet_buffer_writer_factory.end()) {
        return iter->second(media_settings, std::move(header_mem_utils), std::move(payload_mem_utils));
    }
    return nullptr;
}

// Explicit template instantiation
template class RTPMediaPacketBufferWriter<RTPPacketContext>;
template class RTPMediaPacketBufferWriter<RTP_SMPTE_2110_20_PacketContext>;
template class RTPMediaPacketBufferWriter<RTP_SMPTE_2110_40_PacketContext>;
