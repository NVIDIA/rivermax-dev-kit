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

#ifndef RDK_SERVICES_ULP_PACKET_WRITERS_RTP_SMPTE_2110_40_PACKET_WRITER_H_
#define RDK_SERVICES_ULP_PACKET_WRITERS_RTP_SMPTE_2110_40_PACKET_WRITER_H_

#include "rdk/services/ulp_packet/writers/rtp_packet_writer.h"
#include "rdk/services/ulp_packet/rtp_smpte_2110_40_packet_context.h"
#include "rdk/services/media/ancillary_metadata.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

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
 * @brief: RTP packet writer for SMPTE 2110-40 ancillary data.
 *
 * This class provides methods to build RTP headers and fill payloads specific to
 * SMPTE 2110-40 ancillary data packets.
 */
class RTP_SMPTE_2110_40_PacketWriter : public RTPPacketWriter
{
protected:
    AncillaryDataPacketWriter m_ancillary_data_packet_writer;
public:
    /**
     * @brief: Constructor for RTP_SMPTE_2110_40_PacketWriter.
     *
     * Initializes the packet with header and optional payload memory pointers.
     * The payload pointer is optional and used when Header Data Split mode is enabled.
     *
     * @param [in] header_ptr: Pointer to the header memory.
     * @param [in] payload_ptr: Pointer to the payload memory (optional).
     */
    RTP_SMPTE_2110_40_PacketWriter(byte_t* header_ptr, byte_t* payload_ptr);
    virtual ~RTP_SMPTE_2110_40_PacketWriter() = default;
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

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_ULP_PACKET_WRITERS_RTP_SMPTE_2110_40_PACKET_WRITER_H_ */
