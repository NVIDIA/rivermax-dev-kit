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

#ifndef RDK_SERVICES_ULP_PACKET_BUFFER_WR_WRITERS_RTP_MEDIA_PACKET_BUFFER_WRITER_H_
#define RDK_SERVICES_ULP_PACKET_BUFFER_WR_WRITERS_RTP_MEDIA_PACKET_BUFFER_WRITER_H_

#include "rdk/services/media/media_essence_provider.h"
#include "rdk/services/ulp_packet_buffer_wr/writers/ulp_packet_buffer_writer_interface.h"
#include "rdk/services/ulp_packet_buffer_wr/common/rtp_packet.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

constexpr uint32_t DEFAULT_SSRC = 0x0eb51dbd;

/**
 * @brief: Key for media buffer factory map.
 *
 * This struct represents the key used in the media buffer factory map
 * @ref rtp_media_packet_buffer_writer_factory_map_t.
 * It consists of a @ref SMPTEStandard and a boolean indicating whether the buffer contains payload.
 */
struct MediaBufferFactoryKey {
    SMPTEStandard smpte_standard;
    bool contains_payload;
    /**
     * @brief: Equality operator for MediaBufferFactoryKey.
     *
     * @param [in] other: The other MediaBufferFactoryKey to compare with.
     *
     * @return: True if both keys are equal, false otherwise.
     */
    bool operator==(const MediaBufferFactoryKey& other) const {
        return smpte_standard == other.smpte_standard && contains_payload == other.contains_payload;
    }
    /**
     * @brief: Constructor for MediaBufferFactoryKey.
     *
     * @param [in] _smpte_standard: The SMPTE standard.
     * @param [in] _contains_payload: Boolean indicating whether the buffer contains payload.
     */
    MediaBufferFactoryKey(SMPTEStandard _smpte_standard, bool _contains_payload) : smpte_standard(_smpte_standard), contains_payload(_contains_payload) {}
};

/**
 * @brief: Hash function for @ref MediaBufferFactoryKey.
 *
 * This struct provides a hash function for @ref MediaBufferFactoryKey to be used in unordered_map.
 */
struct MediaBufferFactoryKeyHash {
    /**
     * @brief: Hashing operator for MediaBufferFactoryKey.
     *
     * @param [in] key: The MediaBufferFactoryKey to hash.
     *
     * @return: The hash value of the key.
     */
    std::size_t operator()(const MediaBufferFactoryKey& key) const {
        return std::hash<std::underlying_type_t<SMPTEStandard>>()(
            static_cast<std::underlying_type_t<SMPTEStandard>>(key.smpte_standard))
            ^ (std::hash<bool>()(key.contains_payload) << 1);
    }
};

/**
 * @brief: Factory map type for creating @ref RTPMediaPacketBufferWriter instances.
 *
 * This map associates @ref SMPTEStandard values with factory functions that create
 * instances of @ref RTPMediaPacketBufferWriter or its derived classes.
 */
typedef std::unordered_map<
    MediaBufferFactoryKey,
    std::function<std::unique_ptr<IULPPacketBufferWriter>(const MediaSettings& media_settings,
        std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils)>,
        MediaBufferFactoryKeyHash> rtp_media_packet_buffer_writer_factory_map_t;

namespace factory {
    /**
     * @brief: Factory function to create RTP media buffer writers.
     *
     * @param [in] type: SMPTE standard type.
     * @param [in] contains_payload: Flag indicating whether the buffer contains payload.
     * @param [in] media_settings: Media settings.
     * @param [in] header_mem_utils: Shared pointer to header memory utilities.
     * @param [in] payload_mem_utils: Shared pointer to payload memory utilities.
     *
     * @return: Unique pointer to @ref IULPPacketBufferWriter instance.
     */
    std::unique_ptr<IULPPacketBufferWriter> create_rtp_media_packet_buffer_writer(
        SMPTEStandard type, bool contains_payload, const MediaSettings& media_settings,
        std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils);
}

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
template<typename PacketContextType>
class RTPMediaPacketBufferWriter : public IULPPacketBufferWriter
{
protected:
    const MediaSettings& m_media_settings;
    std::unique_ptr<PacketContextType> m_rtp_packet_context;

public:
    /**
     * @brief: Destructor for @ref RTPMediaPacketBufferWriter.
     */
    virtual ~RTPMediaPacketBufferWriter() = default;
    /**
     * @brief: Writes a buffer to RTP packets when Header Data Split mode is off.
     *
     * @param [in] payload_ptr: Pointer to the payload memory.
     * @param [in] length_in_strides: Length of the buffer in strides.
     *
     * @return: Status of the operation.
     */
    ReturnStatus write_buffer(void* payload_ptr, size_t length_in_strides) override;
    /**
     * @brief: Writes a buffer to RTP packets when Header Data Split mode is on.
     *
     * @param [in] header_ptr: Pointer to the header memory.
     * @param [in] payload_ptr: Pointer to the payload memory.
     * @param [in] length_in_strides: Length of the buffer in strides.
     *
     * @return: Status of the operation.
     */
    ReturnStatus write_buffer(void* header_ptr, void* payload_ptr, size_t length_in_strides) override;
    /**
     * @brief: Sets the next media unit to be processed.
     *
     * @param [in] media_unit: Pointer to the media unit (video frame, audio sample, or ancillary data).
     *
     * @return: Return status of the operation.
     */
    virtual ReturnStatus set_next_media_unit(std::shared_ptr<MediaUnit> unit) { return ReturnStatus::success; };
    /**
     * @brief: Sets the rtp timestamp for the first packet.
     *
     * @param [in] packet_time_ns: The timestamp of the first packet.
     */
    void set_initial_timestamp(uint64_t packet_time_ns) override;

protected:
    /**
     * @brief: Constructor for @ref RTPMediaPacketBufferWriter.
     *
     * @param [in] media_settings: Media settings.
     * @param [in] header_mem_utils: Shared pointer to header memory utilities.
     * @param [in] payload_mem_utils: Shared pointer to payload memory utilities.
     */
    RTPMediaPacketBufferWriter(const MediaSettings& media_settings,
        std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils);
    /**
     * @brief: Updates the in-media-unit state.
     *
     * @param [in] header_size: Size of the processed header.
     * @param [in] payload_size: Size of the processed payload.
     */
    virtual void update_in_media_unit_state(size_t header_size, size_t payload_size) = 0;
    /**
     * @brief: Creates the appropriate packet type for this writer.
     *
     * @param [in] header_ptr: Pointer to the header buffer.
     * @param [in] payload_ptr: Pointer to the payload buffer (optional for non-HDS mode).
     *
     * @return: Unique pointer to the created packet.
     */
    virtual std::unique_ptr<RTPPacket> create_packet(byte_t* header_ptr, byte_t* payload_ptr = nullptr) {
        return std::make_unique<RTPPacket>(header_ptr, payload_ptr);
    }
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_ULP_PACKET_BUFFER_WR_WRITERS_RTP_MEDIA_PACKET_BUFFER_WRITER_H_ */
