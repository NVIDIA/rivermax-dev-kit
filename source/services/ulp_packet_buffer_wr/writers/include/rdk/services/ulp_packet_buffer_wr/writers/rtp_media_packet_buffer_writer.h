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

#include <cstddef>
#include <memory>

#include "rdk/services/media/media_essence_provider.h"
#include "rdk/services/ulp_packet_buffer_wr/writers/ulp_packet_buffer_writer_interface.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

constexpr uint32_t DEFAULT_SSRC = 0x0eb51dbd;

/**
 * @brief: RTP send data statistics.
 *
 * This struct will hold run time state of a stream.
 */
struct RTPStreamSendStats
{
    uint32_t packet_counter = 0;
    uint32_t rtp_sequence = 0;
    uint32_t rtp_timestamp = 0;
    uint8_t rtp_interlace_field_indicator = 0;
    uint16_t line_number = 0;
    uint16_t srd_offset = 0;
};

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
class RTPMediaPacketBufferWriter;
typedef std::unordered_map<
    MediaBufferFactoryKey,
    std::function<std::unique_ptr<RTPMediaPacketBufferWriter>(const MediaSettings& media_settings,
        std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils)>,
        MediaBufferFactoryKeyHash> rtp_media_packet_buffer_writer_factory_map_t;

/**
 * @brief: Writes RTP packets with media payload.
 *
 * This class serves as a base class for classes that are responsible for
 * writing RTP packets with media payload. It provides a generic method that
 * fills a provided buffer with a valid RTP header and payload, processes
 * the data, manages the in-frame state logic. Derived classes must implement
 * the pure virtual methods to build the RTP header, write payload,
 * update the in-frame state, and set the concrete stream properties.
 */
class RTPMediaPacketBufferWriter : public IULPPacketBufferWriter
{
protected:
    const MediaSettings& m_media_settings;
    uint32_t m_ssrc = 0;
    RTPStreamSendStats m_send_data;
private:
    /* Factory map for creating RTPMediaPacketBufferWriter instances. */
    static rtp_media_packet_buffer_writer_factory_map_t s_rtp_media_packet_buffer_writer_factory;
public:
    /**
     * @brief: Destructor for @ref RTPMediaPacketBufferWriter.
     */
    virtual ~RTPMediaPacketBufferWriter() = default;
    /**
     * @brief: Factory method to get an @ref RTPMediaPacketBufferWriter instance.
     *
     * @param [in] smpte_standard: SMPTE standard.
     * @param [in] contains_payload: Flag indicating whether the buffer contains payload.
     * @param [in] media_settings: Media settings.
     * @param [in] header_mem_utils: Shared pointer to header memory utilities.
     * @param [in] payload_mem_utils: Shared pointer to payload memory utilities.
     *
     * @return: Unique pointer to an RTPMediaPacketBufferWriter instance.
     */
    static std::unique_ptr<RTPMediaPacketBufferWriter> get_rtp_media_packet_buffer_writer(
        SMPTEStandard smpte_standard, bool contains_payload, const MediaSettings& media_settings,
        std::shared_ptr<MemoryUtils> header_mem_utils, std::shared_ptr<MemoryUtils> payload_mem_utils);
    ReturnStatus write_buffer(void* payload_ptr, size_t length_in_strides) override;
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
    void set_first_packet_timestamp(uint64_t packet_time_ns);
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
     * @brief: Builds the complete RTP header.
     *
     * @param [in] buffer: Pointer to the buffer where the RTP header will be written.
     *
     * @return: The total size of the RTP header and extension written.
     */
    virtual size_t build_rtp_header(byte_t* buffer) = 0;
    /**
     * @brief: Builds the common RTP header.
     *
     * @param [in] buffer: Pointer to the buffer where the RTP header will be written.
     *
     * @return: The size of the RTP header written.
     */
    size_t build_rtp_header_common(byte_t* buffer);
    /**
     * @brief: Fills packet buffer with data.
     *
     * By default, no data is written.
     *
     * @param [in] buffer: Pointer to the buffer where the data will be written.
     *
     * @return: The size of the data written.
     */
    virtual size_t fill_packet(byte_t* buffer) { return 0; };
    /**
     * @brief: Sets the stream properties.
     */
    virtual void set_stream_properties() {};
    /**
     * @brief: Updates the in-media unit state.
     */
    virtual void update_in_media_unit_state() = 0;
    /**
     * @brief: Returns status of Header-Data-Split mode.
     *
     * @return: true if Header-Data-Split mode is enabled.
     */
    inline bool is_hds_on() const { return m_media_settings.app_header_stride_size > 0; }
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_ULP_PACKET_BUFFER_WR_WRITERS_RTP_MEDIA_PACKET_BUFFER_WRITER_H_ */
