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

#include <cstddef>
#include <memory>

#include "rdk/services/media/media_unit_pool.h"
#include "rdk/services/ulp_packet_buffer_wr/writers/rtp_media_packet_buffer_writer.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Writes RTP packets with video payload.
 *
 * This class serves as a mock implementation for writing RTP packets with video payload.
 * It provides methods to set stream properties, update in-frame state, and build RTP headers.
 */
class RTP_SMPTE_2110_20_MockPacketBufferWriter : public RTPMediaPacketBufferWriter
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
    ReturnStatus set_next_media_unit(std::shared_ptr<MediaUnit> unit) override;
protected:
    void set_stream_properties() override {};
    void update_in_media_unit_state() override;
    size_t build_rtp_header(byte_t* buffer) override;
    /**
     * @brief: Builds SMPTE ST 2110-20 extension RTP header.
     *
     * @param [in] buffer: Pointer to the buffer where the extension header will be written.
     *
     * @return: The size of the extension header written.
     */
    size_t build_rtp_header_2110_20_extension(byte_t* buffer);
    /**
     * @brief: Resets the in-media unit state.
     */
    void reset_in_media_unit_state();
};

/**
 * @brief: Writes RTP packets with video payload.
 *
 * This class serves as an implementation for writing RTP packets with video payload.
 * It extends RTP_SMPTE_2110_20_MockPacketBufferWriter and provides additional methods to fill packet buffers.
 */
class RTP_SMPTE_2110_20_PacketBufferWriter : public RTP_SMPTE_2110_20_MockPacketBufferWriter
{
protected:
    size_t m_data_left_in_frame = 0;
    std::shared_ptr<MediaUnit> m_current_media_unit = nullptr;
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
    ReturnStatus set_next_media_unit(std::shared_ptr<MediaUnit> media_unit) override;

    using RTPMediaPacketBufferWriter::write_buffer;
    ReturnStatus write_buffer(void* header_ptr, void* payload_ptr, size_t length_in_strides) override;
protected:
    size_t fill_packet(byte_t* buffer) override;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_ULP_PACKET_BUFFER_WR_WRITERS_RTP_SMPTE_2110_20_PACKET_BUFFER_WRITER_H_ */
