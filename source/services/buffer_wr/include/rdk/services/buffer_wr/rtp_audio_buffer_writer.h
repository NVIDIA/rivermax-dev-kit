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

#ifndef RDK_SERVICES_BUFFER_WR_RTP_AUDIO_BUFFER_WRITER_H_
#define RDK_SERVICES_BUFFER_WR_RTP_AUDIO_BUFFER_WRITER_H_

#include "rdk/services/buffer_wr/rtp_media_buffer_writer.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Simple audio buffer writer for ST 2110-30 RTP packets.
 *
 * @note: This is a simplified implementation for interim use.
 * A more comprehensive refactor is planned for the future.
 */
class RTPAudioBufferWriter : public RTPMediaBufferWriter
{
public:
    RTPAudioBufferWriter(const MediaSettings& media_settings,
        std::shared_ptr<MemoryUtils> header_mem_utils,
        std::shared_ptr<MemoryUtils> payload_mem_utils);
    virtual ~RTPAudioBufferWriter() = default;
    /**
     * @brief: Set the next media unit (audio frame/packet group).
     *
     * @param [in] media_unit: Pointer to the media unit (audio sample).
     *
     * @return: Return status of the operation.
     */
    ReturnStatus set_next_media_unit(std::shared_ptr<MediaUnit> media_unit) override;
protected:
    /**
     * @brief: Build ST 2110-30 RTP header (no extension for audio).
     *
     * @param [in] buffer: Pointer to the buffer where the RTP header will be written.
     *
     * @return: The size of the RTP header written.
     */
    size_t build_rtp_header(byte_t* buffer) override;
    /**
     * @brief: Update packet counter and RTP state for audio.
     */
    void update_in_media_unit_state() override;
    /**
     * @brief: Reset in-media unit state for new media unit.
     */
    void reset_in_media_unit_state();
    /**
     * @brief: Fill packet with audio payload data (for mock implementation, does nothing).
     *
     * @param [in] buffer: Pointer to the buffer where the data will be written.
     *
     * @return: The size of the data written.
     */
    size_t fill_packet(byte_t* buffer) override;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif // RDK_SERVICES_BUFFER_WR_RTP_AUDIO_BUFFER_WRITER_H_
