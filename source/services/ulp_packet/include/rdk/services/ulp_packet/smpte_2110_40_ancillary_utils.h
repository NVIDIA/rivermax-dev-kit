/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef RDK_SERVICES_ULP_PACKET_SMPTE_2110_40_ANCILLARY_UTILS_H_
#define RDK_SERVICES_ULP_PACKET_SMPTE_2110_40_ANCILLARY_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "rdk/services/media/media_defs.h"

namespace rdk
{
namespace services
{

/**
 * @brief: Shared encoding/decoding utilities for SMPTE ST 2110-40 / RFC 8331 ancillary sub-packets.
 */
class AncillaryDataUtils
{
public:
    AncillaryDataUtils() = delete;

    /**
     * @brief: Calculates even parity for a byte.
     *
     * @param [in] value: Byte value to calculate parity for.
     *
     * @return: Even parity bit (0 or 1).
     */
    static inline uint8_t calculate_even_parity(uint8_t value)
    {
        value ^= value >> 4;
        value ^= value >> 2;
        value ^= value >> 1;
        return value & 1;
    }
    /**
     * @brief: Returns bits[7:0] of a 10-bit ST 291-1 word, stripping the two parity bits.
     *
     * @param [in] word: 10-bit ST 291-1 word (parity + data).
     *
     * @return: Data byte (bits 7:0).
     */
    static inline uint8_t strip_parity_bits(uint16_t word)
    {
        return static_cast<uint8_t>(word & MASK_8BIT);
    }
    /**
     * @brief: Returns true if bit[8]=even_parity(bits[7:0]) and bit[9]=NOT(bit[8]).
     *
     * @param [in] word: 10-bit ST 291-1 word to verify.
     *
     * @return: true if parity is valid, false otherwise.
     */
    static inline bool verify_parity_bits(uint16_t word)
    {
        uint8_t data = static_cast<uint8_t>(word & MASK_8BIT);
        const uint8_t parity = calculate_even_parity(data);
        const uint8_t p8 = (word >> RTP_ST_2110_40_PARITY_BIT_POSITION) & 1;
        const uint8_t p9 = (word >> RTP_ST_2110_40_INVERSE_PARITY_BIT_POSITION) & 1;
        return (p8 == parity) && (p9 == (parity ^ 1));
    }
    /**
     * @brief: Adds parity bits to a byte to form a 10-bit ST 291-1 word.
     *
     * @param [in] data: Byte value to add parity bits to.
     *
     * @return: 10-bit word with bit[8]=even_parity(data) and bit[9]=NOT(bit[8]).
     */
    static inline uint16_t add_parity_bits(uint8_t data)
    {
        const uint8_t parity = calculate_even_parity(data);
        return (data & MASK_8BIT)
            | (static_cast<uint16_t>(!parity) << RTP_ST_2110_40_INVERSE_PARITY_BIT_POSITION)
            | (static_cast<uint16_t>(parity)  << RTP_ST_2110_40_PARITY_BIT_POSITION);
    }
    /**
     * @brief: Extracts word_count 10-bit words starting at start_word_index from a packed MSB-first byte stream.
     *
     * @param [in] buffer: Start of the packed byte stream.
     * @param [in] start_word_index: Zero-based index of the first word to unpack.
     * @param [in] word_count: Number of 10-bit words to unpack.
     * @param [out] words: Destination array; must hold at least word_count elements.
     */
    static inline void unpack_10bit_words(const uint8_t* buffer, size_t start_word_index,
        size_t word_count, uint16_t* words)
    {
        size_t bit_pos = start_word_index * RTP_ST_2110_40_DATA_WORD_SIZE_BITS;
        for (size_t i = 0; i < word_count; ++i) {
            const size_t byte_pos = bit_pos / BYTE_SIZE_BITS;
            const size_t bit_offset = bit_pos % BYTE_SIZE_BITS;
            const uint16_t two_bytes = (static_cast<uint16_t>(buffer[byte_pos]) << BYTE_SIZE_BITS)
                | static_cast<uint16_t>(buffer[byte_pos + 1]);
            words[i] = (two_bytes >> (2 * BYTE_SIZE_BITS - RTP_ST_2110_40_DATA_WORD_SIZE_BITS - bit_offset))
                & static_cast<uint16_t>(MASK_10BIT);
            bit_pos += RTP_ST_2110_40_DATA_WORD_SIZE_BITS;
        }
    }
    /**
     * @brief: Packs 10-bit words into a byte buffer, MSB-first, padded to a 32-bit boundary.
     *
     * @param [in] words: Pointer to array of 10-bit words.
     * @param [in] word_count: Number of 10-bit words.
     * @param [out] buffer: Pointer to output byte buffer.
     *
     * @return: Number of bytes written to the buffer.
     */
    static inline size_t pack_10bit_words(const uint16_t* words, size_t word_count, uint8_t* buffer)
    {
        const size_t total_bits = word_count * RTP_ST_2110_40_DATA_WORD_SIZE_BITS;
        const size_t padding_bits = (WORD_SIZE_BITS - (total_bits % WORD_SIZE_BITS)) % WORD_SIZE_BITS;
        const size_t size_in_bytes = (total_bits + padding_bits) / BYTE_SIZE_BITS;
        memset(buffer, 0, size_in_bytes);
        size_t bit_pos = 0;
        size_t leftovers = BYTE_SIZE_BITS;
        for (size_t i = 0; i < word_count; ++i) {
            const uint16_t word = words[i] & MASK_10BIT;
            const size_t byte_pos = bit_pos / BYTE_SIZE_BITS;
            const size_t shift_bits = RTP_ST_2110_40_DATA_WORD_SIZE_BITS - leftovers;
            buffer[byte_pos]     |= (word >> shift_bits) & MASK_8BIT;
            buffer[byte_pos + 1] |= (word << (BYTE_SIZE_BITS - shift_bits)) & MASK_8BIT;
            leftovers = (BYTE_SIZE_BITS - shift_bits) % BYTE_SIZE_BITS;
            if (leftovers == 0) {
                leftovers = BYTE_SIZE_BITS;
            }
            bit_pos += RTP_ST_2110_40_DATA_WORD_SIZE_BITS;
        }
        return size_in_bytes;
    }
    /**
     * @brief: Computes the RFC 8331 10-bit checksum word for an array of packed 10-bit words.
     *
     * Sum of bits[8:0] of each word, modulo 2^9; bit[9] of the result = NOT(bit[8] of sum).
     *
     * @param [in] words: Array of 10-bit packed words (with parity bits included).
     * @param [in] word_count: Number of words.
     *
     * @return: 10-bit checksum word.
     */
    static inline uint16_t calculate_checksum(const uint16_t* words, size_t word_count)
    {
        uint16_t sum = 0;
        for (size_t i = 0; i < word_count; ++i) {
            sum = (sum + (words[i] & static_cast<uint16_t>(MASK_9BIT))) & static_cast<uint16_t>(MASK_9BIT);
        }
        const uint8_t inverse_bit = ((sum >> RTP_ST_2110_40_CHECKSUM_MSB_POSITION) & 1) ? 0 : 1;
        return (static_cast<uint16_t>(inverse_bit) << RTP_ST_2110_40_CHECKSUM_INVERSE_BIT_POSITION)
            | (sum & static_cast<uint16_t>(MASK_9BIT));
    }
    /**
     * @brief: Computes the wire byte size of one ANC sub-packet.
     *
     * @param [in] user_data_words_count: Number of user data words (DATA_COUNT field value).
     *
     * @return: Total byte size of the sub-packet on the wire.
     */
    static inline size_t calculate_packet_size(uint16_t user_data_words_count)
    {
        const size_t packet_word_count = RTP_ST_2110_40_DID_WORD_COUNT
            + RTP_ST_2110_40_SDID_WORD_COUNT
            + RTP_ST_2110_40_DATA_COUNT_WORD_COUNT
            + user_data_words_count;
        const size_t ancillary_packet_bits = packet_word_count * RTP_ST_2110_40_DATA_WORD_SIZE_BITS
            + RTP_ST_2110_40_CHECKSUM_SIZE_BITS;
        const size_t ancillary_packet_padding_bits =
            (WORD_SIZE_BITS - (ancillary_packet_bits % WORD_SIZE_BITS)) % WORD_SIZE_BITS;
        const size_t ancillary_data_packet_payload_bytes =
            (ancillary_packet_bits + ancillary_packet_padding_bits) / BYTE_SIZE_BITS;
        return ancillary_data_packet_payload_bytes + RTP_ST_2110_40_FIXED_DATA_HEADER_SIZE;
    }
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_ULP_PACKET_SMPTE_2110_40_ANCILLARY_UTILS_H_ */
