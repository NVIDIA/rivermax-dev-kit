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

#ifndef RDK_SERVICES_BUFFER_WR_RTP_VIDEO_BUFFER_WRITER_H_
#define RDK_SERVICES_BUFFER_WR_RTP_VIDEO_BUFFER_WRITER_H_

#include <cstddef>
#include <memory>

#include "rdk/services/media/media_frame_provider.h"
#include "rdk/services/buffer_wr/buffer_writer_interface.h"
#include "rdk/services/buffer_wr/rtp_media_buffer_writer.h"

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
class RTPVideoMockBufferWriter : public RTPMediaBufferWriter
{
public:
    /**
     * @brief: Constructor for RTPVideoMockBufferWriter.
     *
     * @param [in] media_settings: Media settings.
     * @param [in] app_header_stride_size: Size of the application header stride.
     * @param [in] data_stride_size: Size of the data stride.
     * @param [in] packet_payload_size: Size of the packet payload.
     * @param [in] header_mem_utils: Shared pointer to header memory utilities.
     * @param [in] payload_mem_utils: Shared pointer to payload memory utilities.
     */
    RTPVideoMockBufferWriter(const MediaSettings& media_settings,
        size_t app_header_stride_size, size_t data_stride_size, uint16_t packet_payload_size,
        std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils);
    /**
     * @brief: Destructor for RTPVideoMockBufferWriter.
     */
    virtual ~RTPVideoMockBufferWriter() = default;
    ReturnStatus set_next_frame(std::shared_ptr<MediaFrame> frame) override;
protected:
    void set_stream_properties() override {};
    void update_in_frame_state() override;
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
     * @brief: Resets the in-frame state.
     */
    void reset_in_frame_state();
};

/**
 * @brief: Writes RTP packets with video payload.
 *
 * This class serves as an implementation for writing RTP packets with video payload.
 * It extends RTPVideoMockBufferWriter and provides additional methods to fill packet buffers.
 */
class RTPVideoBufferWriter : public RTPVideoMockBufferWriter
{
protected:
    size_t m_data_left_in_frame = 0;
    std::shared_ptr<MediaFrame> m_current_frame = nullptr;
public:
    /**
     * @brief: Constructor for RTPVideoBufferWriter.
     *
     * @param [in] media_settings: Media settings.
     * @param [in] app_header_stride_size: Size of the application header stride.
     * @param [in] data_stride_size: Size of the data stride.
     * @param [in] packet_payload_size: Size of the packet payload.
     * @param [in] header_mem_utils: Shared pointer to header memory utilities.
     * @param [in] payload_mem_utils: Shared pointer to payload memory utilities.
     */
    RTPVideoBufferWriter(const MediaSettings& media_settings,
        size_t app_header_stride_size, size_t data_stride_size, uint16_t packet_payload_size,
        std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils) :
        RTPVideoMockBufferWriter(media_settings, app_header_stride_size, data_stride_size,
            packet_payload_size, std::move(header_mem_utils), std::move(payload_mem_utils)) {}
    /**
     * @brief: Destructor for RTPVideoBufferWriter.
     */
    virtual ~RTPVideoBufferWriter() = default;
    ReturnStatus set_next_frame(std::shared_ptr<MediaFrame> frame) override;
protected:
    size_t fill_packet(byte_t* buffer) override;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_BUFFER_WR_RTP_VIDEO_BUFFER_WRITER_H_ */