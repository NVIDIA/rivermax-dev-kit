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

#ifndef RDK_SERVICES_ULP_PACKET_BUFFER_WRITERS_RTP_SMPTE_2110_20_PACKET_BUFFER_WRITER_H_
#define RDK_SERVICES_ULP_PACKET_BUFFER_WRITERS_RTP_SMPTE_2110_20_PACKET_BUFFER_WRITER_H_

#include "rdk/services/ulp_packet_buffer/writers/rtp_media_packet_buffer_writer.h"
#include "rdk/services/ulp_packet/writers/rtp_smpte_2110_20_packet_writer.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Buffer writer for ST 2110-20 RTP packets.
 *
 * This class handles writing RTP packets with video payload.
 */
class RTP_SMPTE_2110_20_PacketBufferWriter : public RTPMediaPacketBufferWriter<RTP_SMPTE_2110_20_PacketContext, RTP_SMPTE_2110_20_PacketWriter>
{
public:
    /**
     * @brief: Constructor for RTP_SMPTE_2110_20_PacketBufferWriter.
     *
     * @param [in] media_settings: Media settings.
     * @param [in] header_mem_utils: Shared pointer to header memory utilities.
     * @param [in] payload_mem_utils: Shared pointer to payload memory utilities.
     * @param [in] enable_zero_copy: Flag to enable zero copy mode.
     *                               If true, only headers are written to the header memory,
     *                               payload is not copied from the media unit to the payload memory.
     */
    RTP_SMPTE_2110_20_PacketBufferWriter(const MediaSettings& media_settings,
        std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils, bool enable_zero_copy);
    /**
     * @brief: Destructor for RTP_SMPTE_2110_20_PacketBufferWriter.
     */
    virtual ~RTP_SMPTE_2110_20_PacketBufferWriter() = default;

protected:
    /** Media unit tracking state */
    uint16_t m_current_line_number = 0;
    uint16_t m_current_srd_offset = 0;
    uint8_t m_current_field_indicator = 0;

    /**
     * @brief: Prepares the packet context for the next packet.
     */
    void prepare_context_for_packet() override;
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
    void reset_in_media_unit_state() override;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_ULP_PACKET_BUFFER_WRITERS_RTP_SMPTE_2110_20_PACKET_BUFFER_WRITER_H_ */
