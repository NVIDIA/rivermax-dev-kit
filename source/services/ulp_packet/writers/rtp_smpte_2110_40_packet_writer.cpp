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

#include <algorithm>
#include <cstring>

#include "rdk/services/ulp_packet/writers/rtp_smpte_2110_40_packet_writer.h"

using namespace rivermax::dev_kit::services;

/**
 * @brief: ST 2110-40 Ancillary RTP extension header.
 *
 * RTP header extension for ancillary data streams following the standard
 * RTP header based on section 2.1 of RFC 8331 - RTP Payload for SMPTE ST
 * 291-1 Ancillary Data.
 */
struct AncillaryRTPExtension
{
    /**
     * @brief: Sets the field indicator value.
     *
     * @param [in] f: Field indicator (0=progressive, 2=field1, 3=field2).
     */
    void set_field_indicator(uint8_t f)
    {
        reserved_byte1 = (f & 0x03) << 6;
    }

    uint16_t extended_sequence_number;
    uint16_t length;
    uint8_t anc_count;
    uint8_t reserved_byte1;
    uint16_t reserved_bytes2_3;
};

RTP_SMPTE_2110_40_PacketWriter::RTP_SMPTE_2110_40_PacketWriter(byte_t* header_ptr, byte_t* payload_ptr)
    : RTPPacketWriter(header_ptr, payload_ptr),
      m_ancillary_data_packet_writer()
{
}

ReturnStatus RTP_SMPTE_2110_40_PacketWriter::fill_header(const IPacketContext& context, size_t& size, MemoryUtils* mem_utils)
{
    const auto& rtp_packet_context = static_cast<const RTP_SMPTE_2110_40_PacketContext&>(context);

    ReturnStatus status = RTPPacketWriter::fill_header(context, size, mem_utils);

    /**
     * @brief: ST 2110-40 Ancillary RTP Header Extension Format.
     *
     * Using extended RTP format based on RFC 8331 - RTP Payload for SMPTE ST 291-1
     * Ancillary Data.
     *
     * 0                   1                   2                   3
     * 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
     * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     * |    Extended Sequence Number   |           Length              |
     * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     * | ANC_Count     |F|   reserved                                  |
     * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     */

    AncillaryRTPExtension* p_ancillary_header = reinterpret_cast<AncillaryRTPExtension*>(m_header_ptr  + size);
    uint16_t high_bits = (rtp_packet_context.extended_sequence_number >> 16) & RTP_SEQUENCE_NUMBER_MASK_16BIT;
    p_ancillary_header->extended_sequence_number = htons(high_bits);

    // Calculate total length and count for all packed descriptors
    uint16_t total_length = 0;
    uint8_t ancillary_packet_count = 0;
    if (rtp_packet_context.descriptors && rtp_packet_context.descriptor_count_in_packet > 0) {
        const std::vector<AncillaryDataDescriptor>& descriptors = *rtp_packet_context.descriptors;
        size_t end_index = std::min(
            rtp_packet_context.descriptor_start_index + rtp_packet_context.descriptor_count_in_packet,
            descriptors.size());

        for (size_t i = rtp_packet_context.descriptor_start_index; i < end_index; ++i) {
            uint16_t ancillary_packet_size = AncillaryDataPacketWriter::calculate_packet_size(
                descriptors[i].ancillary_data_header.user_data_words_count);
            total_length += ancillary_packet_size;
            ancillary_packet_count++;
        }
    }

    p_ancillary_header->length = htons(total_length);
    p_ancillary_header->anc_count = ancillary_packet_count;
    p_ancillary_header->set_field_indicator(rtp_packet_context.field_indicator);

    size = size + sizeof(AncillaryRTPExtension);
    return status;
}

ReturnStatus RTP_SMPTE_2110_40_PacketWriter::fill_payload(const IPacketContext& context, size_t& size, MemoryUtils* mem_utils)
{
    const auto& rtp_packet_context = static_cast<const RTP_SMPTE_2110_40_PacketContext&>(context);

    /**
     * @brief: ST 2110-40 Ancillary RTP Payload Format.
     *
     * Multiple Ancillary data packets can be packed into a single RTP payload
     * based on RFC 8331 - RTP Payload for SMPTE ST 291-1 Ancillary Data.
     *
     *  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     *  |C|   Line_Number=9     |   Horizontal_Offset   |S| StreamNum=0 |
     *  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     *  |         DID       |        SDID       |  Data_Count=0x84  |
     *  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     *                           User_Data_Words...
     *  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     *              |   Checksum_Word   |         word_align            |
     *  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     */

    if (!m_payload_ptr) {
        m_payload_ptr = m_header_ptr + get_header_size();
    }

    size = 0;

    if (rtp_packet_context.payload_ptr == nullptr) {
        size = rtp_packet_context.payload_size;
        return ReturnStatus::success;
    }

    byte_t* current_payload_ptr = m_payload_ptr;
    byte_t* user_data_bytes = rtp_packet_context.payload_ptr;

    if (rtp_packet_context.descriptors && rtp_packet_context.descriptor_count_in_packet > 0) {
        const std::vector<AncillaryDataDescriptor>& descriptors = *rtp_packet_context.descriptors;
        size_t end_index = std::min(
            rtp_packet_context.descriptor_start_index + rtp_packet_context.descriptor_count_in_packet,
            descriptors.size());

        for (size_t i = rtp_packet_context.descriptor_start_index; i < end_index; ++i) {
            const auto& descriptor = descriptors[i];
            uint16_t ancillary_data_packet_size = m_ancillary_data_packet_writer.calculate_packet_size(
                descriptor.ancillary_data_header.user_data_words_count);
            if (size + ancillary_data_packet_size > rtp_packet_context.payload_size) {
                break;
            }

            size_t written = m_ancillary_data_packet_writer.write_ancillary_data(
                current_payload_ptr, user_data_bytes, descriptor);
            current_payload_ptr += written;
            size += written;
        }
    }

    return ReturnStatus::success;
}

size_t RTP_SMPTE_2110_40_PacketWriter::get_header_size() const
{
    return RTPPacketWriter::get_header_size() + sizeof(AncillaryRTPExtension);
}

inline uint8_t AncillaryDataPacketWriter::calculate_even_parity(uint8_t value)
{
    uint8_t parity = 0;
    while (value) {
        parity ^= (value & 1);
        value >>= 1;
    }
    return parity;
}

inline uint16_t AncillaryDataPacketWriter::add_parity_bits(uint8_t data)
{
    uint8_t parity = calculate_even_parity(data);
    return (data & MASK_8BIT) | ((!parity) << RTP_ST_2110_40_INVERSE_PARITY_BIT_POSITION) \
        | (parity << RTP_ST_2110_40_PARITY_BIT_POSITION);
}

inline uint16_t AncillaryDataPacketWriter::calculate_checksum(const std::vector<uint16_t>& packed_words)
{
    uint16_t sum = 0;
    for (size_t i = 0; i < packed_words.size(); ++i) {
        sum = (sum + (packed_words[i] & MASK_9BIT)) & MASK_9BIT; // Ignore carry beyond the 9th bit
    }
    uint8_t inverse_bit = ((sum & MASK_9BIT) >> RTP_ST_2110_40_CHECKSUM_MSB_POSITION) ? 0 : 1;
    return (inverse_bit << RTP_ST_2110_40_CHECKSUM_INVERSE_BIT_POSITION) | (sum & MASK_9BIT);
}

size_t AncillaryDataPacketWriter::pack_10bit_words(const uint16_t* words, size_t word_count, uint8_t* buffer)
{
    // Calculate padding bits required to align total bits to 32-bit word boundaries
    size_t total_bits = word_count * RTP_ST_2110_40_DATA_WORD_SIZE_BITS;
    size_t padding_in_bits = (WORD_SIZE_BITS - (total_bits % WORD_SIZE_BITS)) % WORD_SIZE_BITS;
    size_t size_in_bytes = (total_bits + padding_in_bits) / BYTE_SIZE_BITS;
    // Clear buffer for padding bits
    memset(buffer, 0, size_in_bytes);
    size_t bit_pos = 0;
    size_t leftovers = BYTE_SIZE_BITS;
    for (size_t i = 0; i < word_count; ++i) {
        uint16_t word = words[i] & MASK_10BIT;
        size_t byte_pos = bit_pos / BYTE_SIZE_BITS;
        size_t shift_bits = RTP_ST_2110_40_DATA_WORD_SIZE_BITS - leftovers;
        buffer[byte_pos] |= (word >> shift_bits) & MASK_8BIT;
        buffer[byte_pos + 1] |= (word << (BYTE_SIZE_BITS - shift_bits)) & MASK_8BIT;
        leftovers = (BYTE_SIZE_BITS - shift_bits) % BYTE_SIZE_BITS;
        if (leftovers == 0) {
            leftovers = BYTE_SIZE_BITS;
        }
        bit_pos += RTP_ST_2110_40_DATA_WORD_SIZE_BITS;
    }
    return size_in_bytes;
}

size_t AncillaryDataPacketWriter::write_ancillary_data(byte_t* buffer, byte_t* user_data_bytes,
                                                       const AncillaryDataDescriptor& ancillary_data_descriptor)
{
    uint8_t* ptr = buffer;
    /**
     * @brief: ST 2110-40 ANC data packets Format.
     *  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     *  |C|   Line_Number=9     |   Horizontal_Offset   |S| StreamNum=0 |
     *  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     *  |         DID       |        SDID       |  Data_Count=0x84  |
     *  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     *                           User_Data_Words...
     *  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     *              |   Checksum_Word   |         word_align            |
     *  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     */
    uint32_t word1 = 0;
    word1 |= (ancillary_data_descriptor.ancillary_data_header.c_flag ? 1 : 0) << RTP_ST_2110_40_C_FLAG_BIT_POSITION;
    word1 |= (ancillary_data_descriptor.ancillary_data_header.line_number & MASK_11BIT) << RTP_ST_2110_40_LINE_NUMBER_BIT_POSITION;
    word1 |= (ancillary_data_descriptor.ancillary_data_header.horizontal_offset & MASK_12BIT) << RTP_ST_2110_40_HORIZONTAL_OFFSET_BIT_POSITION;
    word1 |= (ancillary_data_descriptor.ancillary_data_header.s_flag ? 1 : 0) << RTP_ST_2110_40_S_FLAG_BIT_POSITION;
    word1 |= (ancillary_data_descriptor.ancillary_data_header.stream_number & MASK_7BIT);

    // Prepare all the 10-bit words (DID, SDID, Data_Count, User Data and Checksum)
    // and pack them into the buffer, aligned to 32 bits.
    uint16_t did_with_parity = add_parity_bits(ancillary_data_descriptor.ancillary_data_header.did);
    uint16_t sdid_with_parity = add_parity_bits(ancillary_data_descriptor.ancillary_data_header.sdid);
    uint16_t data_count_with_parity = add_parity_bits(static_cast<uint8_t>(ancillary_data_descriptor.ancillary_data_header.user_data_words_count));

    std::vector<uint16_t> packed_words;
    packed_words.push_back(did_with_parity);
    packed_words.push_back(sdid_with_parity);
    packed_words.push_back(data_count_with_parity);
    for (size_t i = 0; i < ancillary_data_descriptor.ancillary_data_header.user_data_words_count; ++i) {
        uint16_t word = add_parity_bits(user_data_bytes[ancillary_data_descriptor.user_data_offset + i]);
        packed_words.push_back(word);
    }

    uint16_t checksum = calculate_checksum(packed_words);
    packed_words.push_back(checksum);

    // Write the first 32-bit word (C, Line Number, Horizontal Offset, S, StreamNum)
    // This is done after reading the user_data_bytes to avoid overwriting the buffer.
    *reinterpret_cast<uint32_t*>(ptr) = htonl(word1);
    ptr += 4;
    // Pack the remaining 10-bit words into the buffer
    ptr += pack_10bit_words(packed_words.data(), packed_words.size(), ptr);

    size_t size_in_bytes = ptr - buffer;
    return size_in_bytes;
}

uint16_t AncillaryDataPacketWriter::calculate_packet_size(uint16_t user_data_words_count)
{
    uint16_t packet_word_count = RTP_ST_2110_40_DID_WORD_COUNT + RTP_ST_2110_40_SDID_WORD_COUNT + RTP_ST_2110_40_DATA_COUNT_WORD_COUNT + user_data_words_count;
    uint16_t ancillary_packet_bits = packet_word_count * RTP_ST_2110_40_DATA_WORD_SIZE_BITS + RTP_ST_2110_40_CHECKSUM_SIZE_BITS;
    uint16_t ancillary_packet_padding_bits = (WORD_SIZE_BITS - (ancillary_packet_bits % WORD_SIZE_BITS)) % WORD_SIZE_BITS;
    uint16_t ancillary_data_packet_payload_bytes = (ancillary_packet_bits + ancillary_packet_padding_bits) / 8;
    return ancillary_data_packet_payload_bytes + RTP_ST_2110_40_FIXED_DATA_HEADER_SIZE;
}
