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

#ifndef RDK_SERVICES_ULP_PACKET_BUFFER_WR_WRITERS_RTP_SMPTE_2110_40_PACKET_BUFFER_WRITER_H_
#define RDK_SERVICES_ULP_PACKET_BUFFER_WR_WRITERS_RTP_SMPTE_2110_40_PACKET_BUFFER_WRITER_H_

#include "rdk/services/ulp_packet_buffer_wr/writers/rtp_media_packet_buffer_writer.h"
#include "rdk/services/ulp_packet_buffer_wr/common/rtp_smpte_2110_40_packet.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Mock buffer writer for ST 2110-40 RTP packets.
 *
 * This class serves as a mock implementation for writing RTP packets with ancillary data.
 * It provides methods to set stream properties, update in-media-unit state, and build RTP headers.
 */
class RTP_SMPTE_2110_40_MockPacketBufferWriter : public RTPMediaPacketBufferWriter<RTP_SMPTE_2110_40_PacketContext>
{
public:
    /**
     * @brief: Constructor for RTP_SMPTE_2110_40_MockPacketBufferWriter.
     *
     * @param [in] media_settings: Media settings.
     * @param [in] header_mem_utils: Shared pointer to header memory utilities.
     * @param [in] payload_mem_utils: Shared pointer to payload memory utilities.
     */
    RTP_SMPTE_2110_40_MockPacketBufferWriter(const MediaSettings& media_settings,
        std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils) :
        RTPMediaPacketBufferWriter<RTP_SMPTE_2110_40_PacketContext>(media_settings, std::move(header_mem_utils), std::move(payload_mem_utils)) {}
    /**
     * @brief: Sets the next media unit (ancillary data packet).
     *
     * @param [in] media_unit: Pointer to the media unit (ancillary data).
     *
     * @return: Return status of the operation.
     */
    ReturnStatus set_next_media_unit(std::shared_ptr<MediaUnit> unit) override;

protected:
    /**
     * @brief: Creates a SMPTE 2110-40 Packet.
     *
     * @param [in] header_ptr: Pointer to the header memory.
     * @param [in] payload_ptr: Pointer to the payload memory (optional).
     *
     * @return: Unique pointer to the created RTP packet.
     */
    std::unique_ptr<RTPPacket> create_packet(byte_t* header_ptr, byte_t* payload_ptr) override {
        return std::make_unique<RTP_SMPTE_2110_40_Packet>(header_ptr, payload_ptr);
    }
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
    void reset_in_media_unit_state();
};

/**
 * @brief: Buffer writer for ST 2110-40 RTP packets.
 *
 * @note This is a simplified implementation for interim use.
 * A more comprehensive refactor is planned for the future.
 */
class RTP_SMPTE_2110_40_PacketBufferWriter : public RTP_SMPTE_2110_40_MockPacketBufferWriter
{
public:
    /**
     * @brief: Constructor for RTP_SMPTE_2110_40_PacketBufferWriter.
     *
     * @param [in] media_settings: Media settings.
     * @param [in] header_mem_utils: Shared pointer to header memory utilities.
     * @param [in] payload_mem_utils: Shared pointer to payload memory utilities.
     */
    RTP_SMPTE_2110_40_PacketBufferWriter(const MediaSettings& media_settings,
        std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils) :
        RTP_SMPTE_2110_40_MockPacketBufferWriter(media_settings, std::move(header_mem_utils), std::move(payload_mem_utils)) {}
    /**
     * @brief: Sets the next media unit (ancillary data packet).
     *
     * @param [in] media_unit: Pointer to the media unit (ancillary data).
     *
     * @return: Return status of the operation.
     */
    ReturnStatus set_next_media_unit(std::shared_ptr<MediaUnit> media_unit) override;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif // RDK_SERVICES_ULP_PACKET_BUFFER_WR_WRITERS_RTP_SMPTE_2110_40_PACKET_BUFFER_WRITER_H_
