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

#ifndef RDK_SERVICES_ULP_PACKET_BUFFER_WRITERS_RTP_MEDIA_PACKET_BUFFER_WRITER_H_
#define RDK_SERVICES_ULP_PACKET_BUFFER_WRITERS_RTP_MEDIA_PACKET_BUFFER_WRITER_H_

#include "rdk/services/media/media_essence_source.h"
#include "rdk/services/ulp_packet_buffer/writers/ulp_packet_buffer_writer_interface.h"
#include "rdk/services/ulp_packet/writers/rtp_packet_writer.h"

namespace rdk
{
namespace services
{

constexpr uint32_t DEFAULT_SSRC = 0x0eb51dbd;

/**
 * @brief: Map type for creating @ref RTPMediaPacketBufferWriter instances.
 *
 * This map associates @ref SMPTEStandard values with factory functions that create
 * instances of @ref RTPMediaPacketBufferWriter or its derived classes.
 */
typedef std::unordered_map<
    SMPTEStandard,
    std::function<std::unique_ptr<IULPPacketBufferWriter>(const MediaSettings& media_settings,
        std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils, bool enable_zero_copy)>> rtp_media_packet_buffer_writer_factory_map_t;

/**
 * @brief: Creates an RTP media packet buffer writer based on the provided parameters.
 *
 * @param [in] smpte_standard: SMPTE standard.
 * @param [in] contains_payload: Flag indicating whether the buffer contains payload.
 * @param [in] media_settings: Media settings.
 * @param [in] header_mem_utils: Shared pointer to header memory utilities.
 * @param [in] payload_mem_utils: Shared pointer to payload memory utilities.
 *
 * @return: Unique pointer to @ref IULPPacketBufferWriter instance.
 */
std::unique_ptr<IULPPacketBufferWriter> create_rtp_media_packet_buffer_writer(
    SMPTEStandard smpte_standard, bool contains_payload, const MediaSettings& media_settings,
    std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils);

/**
 * @brief: Buffer writer for RTP packets.
 *
 * This class serves as a base class for classes that are responsible for
 * writing RTP packets with media payload. It provides a generic method that
 * fills a provided buffer with a valid RTP header and payload, processes
 * the data, manages the in-media-unit state logic. Derived classes must implement
 * the pure virtual methods to build the RTP header, write payload,
 * update the in-media-unit state, and set the concrete stream properties.
 */
template<typename PacketContextType, typename RTPPacketWriterType, typename MetadataType = MediaUnitMetadata>
class RTPMediaPacketBufferWriter : public IULPPacketBufferWriter
{
protected:
    const MediaSettings& m_media_settings;
    std::unique_ptr<PacketContextType> m_rtp_packet_context;
    std::unique_ptr<RTPPacketWriterType> m_rtp_packet_writer;
    bool m_metadata_validated = false;
    bool m_zero_copy_enabled = false;

    /** Media unit tracking state */
    std::shared_ptr<MediaUnit> m_current_media_unit;
    size_t m_bytes_consumed = 0;
    uint32_t m_packet_counter = 0;

public:
    /**
     * @brief: Destructor for @ref RTPMediaPacketBufferWriter.
     */
    virtual ~RTPMediaPacketBufferWriter() = default;
    /**
     * @brief: Writes a buffer to RTP packets when Header Data Split mode is off.
     *
     * @param [in] payload_ptr: Pointer to the payload memory.
     * @param [in] buffer_length: Length of the buffer in strides.
     * @param [in] payload_sizes: Optional array to fill with actual payload sizes if not nullptr.
     *
     * @return: Status of the operation.
     */
    ReturnStatus write_buffer(void* payload_ptr, size_t buffer_length, uint16_t* payload_sizes = nullptr) override;
    /**
     * @brief: Writes a buffer to RTP packets when Header Data Split mode is on.
     *
     * @param [in] header_ptr: Pointer to the header memory.
     * @param [in] payload_ptr: Pointer to the payload memory.
     * @param [in] buffer_length: Length of the buffer in strides.
     * @param [in] header_sizes: Optional array to fill with actual header sizes if not nullptr.
     * @param [in] payload_sizes: Optional array to fill with actual payload sizes if not nullptr.
     *
     * @return: Status of the operation.
     */
    ReturnStatus write_buffer(void* header_ptr, void* payload_ptr, size_t buffer_length,
                              uint16_t* header_sizes = nullptr, uint16_t* payload_sizes = nullptr) override;
    /**
     * @brief: Sets the next media unit to be processed.
     *
     * @param [in] media_unit: Pointer to the media unit (video frame, audio sample, or ancillary data).
     *
     * @return: Return status of the operation.
     */
    ReturnStatus set_next_media_unit(std::shared_ptr<MediaUnit> media_unit) override;
    /**
     * @brief: Sets the RTP timestamp for the first packet.
     *
     * @param [in] time_ns: The timestamp of the first packet.
     */
    void set_start_time(uint64_t time_ns) override;
    /**
     * @brief: Returns the number of packets needed for the next chunk.
     *
     * Returns fixed value from media settings by default.
     *
     * @return: Number of packets for the next chunk.
     */
    size_t get_num_packets_for_next_chunk() const override
    {
        return m_media_settings.packets_in_chunk;
    }

protected:
    /**
     * @brief: Constructor for @ref RTPMediaPacketBufferWriter.
     *
     * @param [in] media_settings: Media settings.
     * @param [in] header_mem_utils: Shared pointer to header memory utilities.
     * @param [in] payload_mem_utils: Shared pointer to payload memory utilities.
     * @param [in] enable_zero_copy: Flag to enable zero copy mode.
     *                               If true, only headers are written to the header memory,
     *                               payload is not copied from the media unit to the payload memory.
     */
    RTPMediaPacketBufferWriter(const MediaSettings& media_settings,
        std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils, bool enable_zero_copy = false);
    /**
     * @brief: Prepares the packet context for the next packet.
     *
     * Called before each @ref fill_header() / @ref fill_payload() to populate context
     * fields from the buffer writer's tracking state.
     * Also sets payload_ptr for payload access.
     */
    virtual void prepare_context_for_packet() = 0;
    /**
     * @brief: Updates the in-media-unit state.
     *
     * @param [in] header_size: Size of the processed header.
     * @param [in] payload_size: Size of the processed payload.
     */
    virtual void update_in_media_unit_state(size_t header_size, size_t payload_size) = 0;
    /**
     * @brief: Reset in-media unit state for new media unit.
     */
    virtual void reset_in_media_unit_state() = 0;
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_ULP_PACKET_BUFFER_WRITERS_RTP_MEDIA_PACKET_BUFFER_WRITER_H_ */
