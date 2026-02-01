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

#ifndef RDK_SERVICES_ULP_PACKET_BUFFER_COMMON_RTP_SMPTE_2110_40_PACKET_H_
#define RDK_SERVICES_ULP_PACKET_BUFFER_COMMON_RTP_SMPTE_2110_40_PACKET_H_

#include "rdk/services/ulp_packet_buffer/common/rtp_packet.h"
#include "rdk/services/media/media.h"
#include "rdk/services/media/ancillary_metadata.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Context for RTP SMPTE 2110-40 packets.
 *
 * This struct extends the @ref RTPPacketContext to include fields
 * specific to SMPTE 2110-40 ancillary data packets based on section 2.1 of
 * RFC 8331 - RTP Payload for SMPTE ST 291-1 Ancillary Data.
 */
struct RTP_SMPTE_2110_40_PacketContext : public RTPPacketContext
{
    uint32_t extended_sequence_number = 0;             /**< 32-bit extended sequence number */
    uint16_t length;                                   /**< Number of octets of the ANC data RTP payload */
    uint32_t ancillary_count = 0;                      /**< Number of ancillary data packets */
    uint8_t field_indicator = 0;                       /**< Field indicator specifying RTP timestamp
                                                            relation to video fields */
    size_t descriptor_start_index = 0;                 /**< Index of first descriptor to pack in this RTP packet */
    size_t descriptor_count_in_packet = 0;             /**< Number of descriptors to pack in this RTP packet */

    /**
     * @brief: Returns the ancillary descriptors from the current media unit metadata.
     *
     * @return: Pointer to ancillary descriptors vector.
     */
    const std::vector<AncillaryDataDescriptor>* get_ancillary_descriptors() const
    {
        if (!current_media_unit || !current_media_unit->metadata) {
            return nullptr;
        }
        auto* ancillary_metadata = static_cast<AncillaryMediaUnitMetadata*>(
            current_media_unit->metadata.get());
        return &ancillary_metadata->ancillary_data;
    }
};

/**
 * @brief: Helper class to write ancillary data payloads into a buffer.
 *
 * Provides methods to write ancillary data packets that are carried in the RTP payload
 * based on section 2.1 of RFC 8331 - RTP Payload for SMPTE ST 291-1 Ancillary Data.
 */
class AncillaryDataPacketWriter
{
private:
    /**
     * @brief: Calculate even parity for a byte.
     *
     * @param [in] value: Byte value to calculate parity for.
     *
     * @return: Even parity bit (0 or 1).
     */
    static uint8_t calculate_even_parity(uint8_t value);
    /**
     * @brief: Adds parity bits to a byte to form a 10-bit word.
     *
     * @param [in] data: Byte value to add parity bits to.
     *
     * @return: 10-bit word with parity bits.
     */
    static uint16_t add_parity_bits(uint8_t data);
    /**
     * @brief: Calculates checksum for ancillary data packet.
     *
     * @param [in] packed_words: Vector of packed 10-bit words.
     *
     * @return: Calculated checksum (9 bits).
     */
    static uint16_t calculate_checksum(const std::vector<uint16_t>& packed_words);
    /**
     * @brief: Packs 10-bit words into a byte buffer.
     *
     * @param [in] words: Pointer to array of 10-bit words.
     * @param [in] word_count: Number of 10-bit words.
     * @param [out] buffer: Pointer to output byte buffer.
     *
     * @return: Number of bytes written to the buffer.
     */
    static size_t pack_10bit_words(const uint16_t* words, size_t word_count, uint8_t* buffer);

public:
    AncillaryDataPacketWriter() {}
    /**
     * @brief: Writes ancillary data packet into the buffer.
     *
     * @param [out] buffer: Pointer to the buffer to write the packet into.
     * @param [in] user_data_bytes: Pointer to array of 8-bit user data words (before parity bits are added).
     * @param [in] ancillary_data_header: Ancillary data header containing ancillary data packet information.
     *
     * @return: Total number of bytes written to the buffer.
     */
    static size_t write_ancillary_data(byte_t* buffer, byte_t* user_data_bytes,
                                       const AncillaryDataDescriptor& ancillary_data_descriptor);
    /**
     * @brief: Calculates the size of the ancillary data packet.
     *
     * @param [in] user_data_words_count: Number of 10-bit words in user data.
     *
     * @return: Size of the ancillary data packet in bytes.
     */
    static uint16_t calculate_packet_size(uint16_t user_data_words_count);
};

/**
 * @brief: RTP packet structure for SMPTE 2110-40 ancillary data.
 *
 * This class provides methods to build RTP headers and fill payloads specific to
 * SMPTE 2110-40 ancillary data packets.
 */
class RTP_SMPTE_2110_40_Packet : public RTPPacket
{
protected:
    AncillaryDataPacketWriter m_ancillary_data_packet_writer;
public:
    /**
     * @brief: Constructor for RTP_SMPTE_2110_40_Packet.
     *
     * Initializes the packet with header and optional payload memory pointers.
     * The payload pointer is optional and used when Header Data Split mode is enabled.
     *
     * @param [in] header_ptr: Pointer to the header memory.
     * @param [in] payload_ptr: Pointer to the payload memory (optional).
     */
    RTP_SMPTE_2110_40_Packet(byte_t* header_ptr, byte_t* payload_ptr);
    virtual ~RTP_SMPTE_2110_40_Packet() = default;
    /**
     * @brief: Fills the RTP SMPTE 2110-40 packet header.
     *
     * @param [in] context: The packet context containing relevant information.
     * @param [out] size: Reference to store the size of the filled header.
     * @param [in] mem_utils: Memory utilities for header manipulation.
     *
     * @return: The status of the operation.
     */
    ReturnStatus fill_header(const IPacketContext& context, size_t& size, MemoryUtils* mem_utils) override;
    /**
     * @brief: Fills the RTP SMPTE 2110-40 packet payload.
     *
     * @param [in] context: The packet context containing relevant information.
     * @param [out] size: Reference to store the size of the filled payload.
     * @param [in] mem_utils: Memory utilities for payload manipulation.
     *
     * @return: The status of the operation.
     */
    ReturnStatus fill_payload(const IPacketContext& context, size_t& size, MemoryUtils* mem_utils) override;
    /**
     * @brief: Returns the size of the RTP SMPTE 2110-40 packet header.
     *
     * @return: The size of the RTP SMPTE 2110-40 packet header in bytes.
     */
    size_t get_header_size() const override;
};

/**
 * @brief: Mock RTP packet structure for SMPTE 2110-40 ancillary data.
 */
class RTP_SMPTE_2110_40_MockPacket : public RTP_SMPTE_2110_40_Packet
{
private:
    // Temporary variables to track data initialization and avoid redundant fills
    // TODO: Refactor this logic and provide a payload that already contains the ancillary data
    void* m_data_ptr;
    bool initialized = false;
public:
    /**
     * @brief: Constructor for RTP_SMPTE_2110_40_MockPacket.
     *
     * Initializes the packet with header and optional payload memory pointers.
     * The payload pointer is optional and used when Header Data Split mode is enabled.
     *
     * @param [in] header_ptr: Pointer to the header memory.
     * @param [in] payload_ptr: Pointer to the payload memory (optional).
     */
    RTP_SMPTE_2110_40_MockPacket(byte_t* header_ptr, byte_t* payload_ptr) :
        RTP_SMPTE_2110_40_Packet(header_ptr, payload_ptr) {
            // Temporary initialization and will be removed in future refactor
            m_data_ptr = payload_ptr;
            initialized = false;
        }
    /**
     * @brief: Fills the RTP SMPTE 2110-40 packet payload.
     *
     * @param [in] context: The packet context containing relevant information.
     * @param [out] size: Reference to store the size of the filled payload.
     * @param [in] mem_utils: Memory utilities for payload manipulation.
     *
     * @return: The status of the operation.
     */
    ReturnStatus fill_payload(const IPacketContext& context, size_t& size, MemoryUtils* mem_utils) override;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_ULP_PACKET_BUFFER_COMMON_RTP_SMPTE_2110_40_PACKET_H_ */
