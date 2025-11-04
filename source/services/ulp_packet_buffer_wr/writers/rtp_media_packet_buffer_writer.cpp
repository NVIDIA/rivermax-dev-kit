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
#include <memory>

#include "rt_threads.h"

#include "rdk/services/ulp_packet_buffer_wr/writers/rtp_media_packet_buffer_writer.h"
#include "rdk/services/media/ancillary_metadata.h"
#include "rdk/services/ulp_packet_buffer_wr/writers/rtp_smpte_2110_20_packet_buffer_writer.h"
#include "rdk/services/ulp_packet_buffer_wr/writers/rtp_smpte_2110_30_packet_buffer_writer.h"
#include "rdk/services/ulp_packet_buffer_wr/writers/rtp_smpte_2110_40_packet_buffer_writer.h"

using namespace rivermax::dev_kit::services;

template<typename PacketContextType, typename RTPPacketType, typename MetadataType>
RTPMediaPacketBufferWriter<PacketContextType, RTPPacketType, MetadataType>::RTPMediaPacketBufferWriter(const MediaSettings& media_settings,
    std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils, bool enable_mock_mode) :
    IULPPacketBufferWriter(std::move(header_mem_utils), std::move(payload_mem_utils)),
    m_media_settings(media_settings)
{
    m_rtp_packet_context = std::make_unique<PacketContextType>();
    m_rtp_packet_context->ssrc = DEFAULT_SSRC; // Simulated SSRC.
    m_rtp_packet_context->payload_type = media_settings.payload_type;
    m_rtp_packet_context->payload_size = media_settings.raw_packet_payload_size;
    m_rtp_packet = std::make_unique<RTPPacketType>(nullptr, nullptr);
    m_mock_mode_enabled = enable_mock_mode;
}

template<typename PacketContextType, typename RTPPacketType, typename MetadataType>
ReturnStatus RTPMediaPacketBufferWriter<PacketContextType, RTPPacketType, MetadataType>::set_next_media_unit(std::shared_ptr<MediaUnit> media_unit)
{
    if (media_unit == nullptr || media_unit->data == nullptr) {
        std::cerr << "Error: Media unit is null or media unit data is null" << std::endl;
        return ReturnStatus::failure;
    }

    // Validate metadata type only once, or until validation succeeds
    if (unlikely(!m_metadata_validated)) {
        m_metadata_ptr = dynamic_cast<const MetadataType*>(media_unit->metadata.get());
        if (!m_metadata_ptr) {
            std::cerr << "Error: Invalid metadata type for buffer writer. "
                      << "Expected: " << typeid(MetadataType).name()
                      << ", Received: " << (media_unit->metadata ? typeid(*media_unit->metadata).name() : "null")
                      << std::endl;
            return ReturnStatus::failure;
        }
        m_metadata_validated = true;
    } else {
        m_metadata_ptr = static_cast<const MetadataType*>(media_unit->metadata.get());
    }
    reset_in_media_unit_state();
    m_rtp_packet_context->current_media_unit = std::move(media_unit);
    m_rtp_packet_context->data_left_in_media_unit_in_bytes = m_rtp_packet_context->current_media_unit->data->get_size();
    return ReturnStatus::success;
}

template<typename PacketContextType, typename RTPPacketType, typename MetadataType>
ReturnStatus RTPMediaPacketBufferWriter<PacketContextType, RTPPacketType, MetadataType>::write_buffer(void* payload_ptr, size_t buffer_length)
{
    byte_t* current_packet_pointer = reinterpret_cast<byte_t*>(payload_ptr);
    assert(current_packet_pointer);
    uint64_t stride = 0;
    size_t header_size = 0;
    size_t payload_size = 0;

    while (stride < buffer_length && m_rtp_packet_context->counter < m_media_settings.packets_in_media_unit) {
        m_rtp_packet->set_packet(current_packet_pointer);
        // Skip ReturnStatus testing for performance reasons
        (void)m_rtp_packet->fill_header(*m_rtp_packet_context, header_size, m_header_mem_utils.get());
        (void)m_rtp_packet->fill_payload(*m_rtp_packet_context, payload_size, m_payload_mem_utils.get());
        update_in_media_unit_state(header_size, payload_size);
        current_packet_pointer += m_media_settings.data_stride_size;
        stride++;
    }
    return ReturnStatus::success;
}

template<typename PacketContextType, typename RTPPacketType, typename MetadataType>
ReturnStatus RTPMediaPacketBufferWriter<PacketContextType, RTPPacketType, MetadataType>::write_buffer(void* header_ptr, void* payload_ptr, size_t buffer_length)
{
    byte_t* current_header_pointer = reinterpret_cast<byte_t*>(header_ptr);
    byte_t* current_payload_pointer = reinterpret_cast<byte_t*>(payload_ptr);
    assert(current_header_pointer);
    assert(current_payload_pointer);
    uint64_t stride = 0;
    size_t header_size = 0;
    ReturnStatus status = ReturnStatus::success;

    while (stride < buffer_length && m_rtp_packet_context->counter < m_media_settings.packets_in_media_unit) {
        m_rtp_packet->set_packet(current_header_pointer, current_payload_pointer); // Header Data Split mode
        status = m_rtp_packet->fill_header(*m_rtp_packet_context, header_size, m_header_mem_utils.get());
        update_in_media_unit_state(header_size, 0);
        current_header_pointer += m_media_settings.app_header_stride_size;
        stride++;
    }

    if (status != ReturnStatus::success) {
        std::cerr << "Failed to fill RTP headers" << std::endl;
        return status;
    }

    if (m_mock_mode_enabled) {
        // Mock mode - Data was pre loaded / No media unit assigned
        return ReturnStatus::success;
    }

    byte_t* media_unit_ptr =
        m_rtp_packet_context->current_media_unit->data->get() + (m_rtp_packet_context->current_media_unit->data->get_size() - \
        m_rtp_packet_context->data_left_in_media_unit_in_bytes);
    status = m_payload_mem_utils->memory_copy_2D(current_payload_pointer, m_media_settings.data_stride_size,
        media_unit_ptr, m_media_settings.raw_packet_payload_size, m_media_settings.raw_packet_payload_size,
        stride, m_rtp_packet_context->current_media_unit->data->get_memory_location());

    size_t data_copied = std::min(
        stride * m_media_settings.raw_packet_payload_size,
        m_rtp_packet_context->data_left_in_media_unit_in_bytes);
    m_rtp_packet_context->data_left_in_media_unit_in_bytes -= data_copied;

    if (status != ReturnStatus::success) {
        std::cerr << "Failed to 2D copy" << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

template<typename PacketContextType, typename RTPPacketType, typename MetadataType>
void RTPMediaPacketBufferWriter<PacketContextType, RTPPacketType, MetadataType>::set_initial_timestamp(uint64_t packet_time_ns)
{
    m_rtp_packet_context->timestamp = static_cast<uint32_t>(
        time_to_rtp_timestamp(packet_time_ns, static_cast<int>(m_media_settings.sample_rate)));
}


template<typename WriterType>
std::unique_ptr<IULPPacketBufferWriter> create_writer(
    const MediaSettings& media_settings,
    std::shared_ptr<MemoryUtils> header_mem_utils,
    std::shared_ptr<MemoryUtils> payload_mem_utils,
    bool enable_mock_mode)
{
    return std::unique_ptr<WriterType>(new WriterType(media_settings,
        std::move(header_mem_utils), std::move(payload_mem_utils), enable_mock_mode));
}

static rtp_media_packet_buffer_writer_factory_map_t s_rtp_media_packet_buffer_writer_factory = {
    {SMPTEStandard::ST_2110_20, create_writer<RTP_SMPTE_2110_20_PacketBufferWriter>},
    {SMPTEStandard::ST_2110_30, create_writer<RTP_SMPTE_2110_30_PacketBufferWriter>},
    {SMPTEStandard::ST_2110_40, create_writer<RTP_SMPTE_2110_40_PacketBufferWriter>}
};

std::unique_ptr<IULPPacketBufferWriter> rivermax::dev_kit::services::create_rtp_media_packet_buffer_writer(
    SMPTEStandard smpte_type, bool contains_payload, const MediaSettings& media_settings,
    std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils)
{
    auto iter = s_rtp_media_packet_buffer_writer_factory.find(smpte_type);
    if (iter != s_rtp_media_packet_buffer_writer_factory.end()) {
        return iter->second(media_settings, std::move(header_mem_utils), std::move(payload_mem_utils), !contains_payload);
    }
    return nullptr;
}

// Explicit template instantiation
template class RTPMediaPacketBufferWriter<RTPPacketContext, RTPPacket, MediaUnitMetadata>;
template class RTPMediaPacketBufferWriter<RTP_SMPTE_2110_20_PacketContext, RTP_SMPTE_2110_20_Packet, MediaUnitMetadata>;
template class RTPMediaPacketBufferWriter<RTPPacketContext, RTP_SMPTE_2110_30_Packet, MediaUnitMetadata>;
template class RTPMediaPacketBufferWriter<RTP_SMPTE_2110_40_PacketContext, RTP_SMPTE_2110_40_Packet, AncillaryMediaUnitMetadata>;
