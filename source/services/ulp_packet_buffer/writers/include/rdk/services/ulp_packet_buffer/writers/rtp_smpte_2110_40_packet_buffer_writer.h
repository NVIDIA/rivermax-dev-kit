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

#ifndef RDK_SERVICES_ULP_PACKET_BUFFER_WRITERS_RTP_SMPTE_2110_40_PACKET_BUFFER_WRITER_H_
#define RDK_SERVICES_ULP_PACKET_BUFFER_WRITERS_RTP_SMPTE_2110_40_PACKET_BUFFER_WRITER_H_

#include "rdk/services/media/ancillary_metadata.h"
#include "rdk/services/ulp_packet_buffer/writers/rtp_media_packet_buffer_writer.h"
#include "rdk/services/ulp_packet/writers/rtp_smpte_2110_40_packet_writer.h"

namespace rdk
{
namespace services
{

/**
 * @brief: Buffer writer for ST 2110-40 RTP packets.
 *
 * This class handles writing RTP packets with ancillary data.
 * Ancillary data header will be taken from specialized metadata.
 */
class RTP_SMPTE_2110_40_PacketBufferWriter : public RTPMediaPacketBufferWriter<RTP_SMPTE_2110_40_PacketContext, RTP_SMPTE_2110_40_PacketWriter, AncillaryMediaUnitMetadata>
{
protected:
    size_t m_cached_packets_in_media_unit = 0;

    /** Media unit tracking state */
    std::vector<AncillaryDataDescriptor>* m_descriptors = nullptr;
    size_t m_current_descriptor_index = 0;
    uint8_t m_current_field_indicator = RTP_2110_40_FIELD_INDICATOR_PROGRESSIVE;

public:
    /**
     * @brief: Constructor for RTP_SMPTE_2110_40_PacketBufferWriter.
     *
     * @param [in] media_settings: Media settings.
     * @param [in] header_mem_utils: Shared pointer to header memory utilities.
     * @param [in] payload_mem_utils: Shared pointer to payload memory utilities.
     * @param [in] enable_zero_copy: Flag to enable zero copy mode.
     *                               If true, only headers are written to the header memory,
     *                               payload is not copied from the media unit to the payload memory.
     */
    RTP_SMPTE_2110_40_PacketBufferWriter(const MediaSettings& media_settings,
        std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils, bool enable_zero_copy);
    /**
     * @brief: Writes RTP packet buffer when Header Data Split mode is off.
     *
     * @param [in] payload_ptr: Pointer to the payload memory (header + payload together).
     * @param [in] buffer_length: Length of the buffer in strides.
     * @param [out] payload_sizes: Optional array to fill with actual payload sizes if not nullptr.
     *
     * @return: Status of the operation.
     */
    ReturnStatus write_buffer(void* payload_ptr, size_t buffer_length, uint16_t* payload_sizes = nullptr) override;
    /**
     * @brief: Writes RTP packet buffer when Header Data Split mode is on.
     *
     * @param [in] header_ptr: Pointer to the header memory.
     * @param [in] payload_ptr: Pointer to the payload memory.
     * @param [in] buffer_length: Length of the buffer in strides.
     * @param [out] header_sizes: Optional array to fill with actual header sizes if not nullptr.
     * @param [out] payload_sizes: Optional array to fill with actual payload sizes if not nullptr.
     *
     * @return: Status of the operation.
     */
    ReturnStatus write_buffer(void* header_ptr, void* payload_ptr, size_t buffer_length, uint16_t* header_sizes = nullptr, uint16_t* payload_sizes = nullptr) override;
    /**
     * @brief: Set the next media unit and populate initial packet context from ancillary metadata.
     *
     * @param [in] media_unit: Pointer to the media unit containing ancillary data.
     *
     * @return: Status of the operation.
     */
     ReturnStatus set_next_media_unit(std::shared_ptr<MediaUnit> media_unit) override;
    /**
     * @brief: Returns the number of packets needed for the next chunk.
     *
     * Returns packets required for the current media unit.
     *
     * @return: Number of packets for the next chunk.
     */
     size_t get_num_packets_for_next_chunk() const override;

protected:
    /**
     * @brief: Prepares the packet context for the next packet.
     */
    void prepare_context_for_packet() override;
    /**
     * @brief: Updates the packet counter and RTP state for ancillary.
     *
     * @param [in] header_size: Size of the processed header.
     * @param [in] payload_size: Size of the processed payload.
     */
    void update_in_media_unit_state(size_t header_size, size_t payload_size) override;
    /**
     * @brief: Reset in-media unit state for new media unit.
     */
    void reset_in_media_unit_state() override;
    /**
     * @brief: Calculates number of packets needed for current media unit.
     *
     * Helper function to compute the number of RTP packets required
     * for the current media unit based on ANC packet count and chunk distribution.
     *
     * @return: Number of packets for the media unit.
     */
    size_t calculate_packets_for_media_unit() const;
    /**
     * @brief: Calculates how many descriptors can fit in one RTP packet starting from given index.
     *
     * @param [in] descriptors: Const reference to vector of ancillary data descriptors.
     * @param [in] start_index: Starting index in the descriptor array.
     * @param [in] max_payload_size: Maximum payload size in bytes.
     * @param [in] max_ancillary_data_packets_per_packet: Maximum number of ancillary data packets per RTP packet.
     *
     * @return: Number of descriptors that can fit.
     */
    static size_t calculate_descriptors_in_packet(
        const std::vector<AncillaryDataDescriptor>& descriptors,
        size_t start_index,
        size_t max_payload_size,
        size_t max_ancillary_data_packets_per_packet);
    /**
     * @brief: Calculates the actual number of RTP packets needed for all descriptors.
     *
     * @param [in] descriptors: Const reference to vector of ancillary data descriptors.
     * @param [in] max_payload_size: Maximum payload size in bytes.
     * @param [in] max_ancillary_data_packets_per_packet: Maximum number of ancillary data packets per RTP packet.
     *
     * @return: Number of RTP packets required.
     */
    static size_t calculate_rtp_packets_for_descriptors(
        const std::vector<AncillaryDataDescriptor>& descriptors,
        size_t max_payload_size,
        size_t max_ancillary_data_packets_per_packet);
};

} // namespace services
} // namespace rdk

#endif // RDK_SERVICES_ULP_PACKET_BUFFER_WRITERS_RTP_SMPTE_2110_40_PACKET_BUFFER_WRITER_H_
