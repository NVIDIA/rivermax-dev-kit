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

#ifndef RDK_SERVICES_ULP_PACKET_BUFFER_WR_WRITERS_RTP_SMPTE_2110_20_PACKET_BUFFER_WRITER_H_
#define RDK_SERVICES_ULP_PACKET_BUFFER_WR_WRITERS_RTP_SMPTE_2110_20_PACKET_BUFFER_WRITER_H_

#include "rdk/services/ulp_packet_buffer_wr/writers/rtp_media_packet_buffer_writer.h"
#include "rdk/services/ulp_packet_buffer_wr/common/rtp_smpte_2110_20_packet.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Mock buffer writer for ST 2110-20 RTP packets.
 *
 * This class serves as a mock implementation for writing RTP packets with video payload.
 * It provides methods to set stream properties, update in-media-unit state, and build RTP headers.
 */
class RTP_SMPTE_2110_20_MockPacketBufferWriter : public RTPMediaPacketBufferWriter<RTP_SMPTE_2110_20_PacketContext>
{
public:
    /**
     * @brief: Constructor for RTP_SMPTE_2110_20_MockPacketBufferWriter.
     *
     * @param [in] media_settings: Media settings.
     * @param [in] header_mem_utils: Shared pointer to header memory utilities.
     * @param [in] payload_mem_utils: Shared pointer to payload memory utilities.
     */
    RTP_SMPTE_2110_20_MockPacketBufferWriter(const MediaSettings& media_settings,
        std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils);
    /**
     * @brief: Destructor for RTP_SMPTE_2110_20_MockPacketBufferWriter.
     */
    virtual ~RTP_SMPTE_2110_20_MockPacketBufferWriter() = default;
    /**
     * @brief: Sets the next media unit (video frame).
     *
     * @param [in] media_unit: Pointer to the media unit.
     *
     * @return: Return status of the operation.
     */
    ReturnStatus set_next_media_unit(std::shared_ptr<MediaUnit> unit) override;

protected:
    /**
     * @brief: Creates a SMPTE 2110-20 Packet.
     *
     * @param [in] header_ptr: Pointer to the header memory.
     * @param [in] payload_ptr: Pointer to the payload memory (optional).
     *
     * @return: Unique pointer to the created RTP packet.
     */
    std::unique_ptr<RTPPacket> create_packet(byte_t* header_ptr, byte_t* payload_ptr) override {
        return std::make_unique<RTP_SMPTE_2110_20_Packet>(header_ptr, payload_ptr);
    }
    /**
     * @brief: Updates the in-media-unit state.
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
 * @brief: Buffer writer for ST 2110-20 RTP packets.
 *
 * This class serves as an implementation for writing RTP packets with video payload.
 * It extends RTP_SMPTE_2110_20_MockPacketBufferWriter and provides additional methods to write media to buffers.
 */
class RTP_SMPTE_2110_20_PacketBufferWriter : public RTP_SMPTE_2110_20_MockPacketBufferWriter
{
public:
    /**
     * @brief: Constructor for RTP_SMPTE_2110_20_PacketBufferWriter.
     *
     * @param [in] media_settings: Media settings.
     * @param [in] header_mem_utils: Shared pointer to header memory utilities.
     * @param [in] payload_mem_utils: Shared pointer to payload memory utilities.
     */
    RTP_SMPTE_2110_20_PacketBufferWriter(const MediaSettings& media_settings,
        std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils) :
        RTP_SMPTE_2110_20_MockPacketBufferWriter(media_settings, std::move(header_mem_utils), std::move(payload_mem_utils)) {}
    /**
     * @brief: Destructor for RTP_SMPTE_2110_20_PacketBufferWriter.
     */
    virtual ~RTP_SMPTE_2110_20_PacketBufferWriter() = default;
    /**
     * @brief: Sets the next media unit (video frame).
     *
     * @param [in] media_unit: Pointer to the media unit.
     *
     * @return: Return status of the operation.
     */
    ReturnStatus set_next_media_unit(std::shared_ptr<MediaUnit> media_unit) override;
    // Inherit the other overload of write_buffer
    using RTPMediaPacketBufferWriter::write_buffer;
    /**
     * @brief: Writes a buffer to RTP packets.
     *
     * @param [in] header_ptr: Pointer to the header memory.
     * @param [in] payload_ptr: Pointer to the payload memory.
     * @param [in] length_in_strides: Length of the buffer in strides.
     *
     * @return: Status of the operation.
     */
    ReturnStatus write_buffer(void* header_ptr, void* payload_ptr, size_t length_in_strides) override;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_ULP_PACKET_BUFFER_WR_WRITERS_RTP_SMPTE_2110_20_PACKET_BUFFER_WRITER_H_ */
