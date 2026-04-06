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

#ifndef RDK_SERVICES_ULP_PACKET_RTP_SMPTE_2110_20_EXTENSION_H_
#define RDK_SERVICES_ULP_PACKET_RTP_SMPTE_2110_20_EXTENSION_H_

#include <cstdint>

namespace rdk
{
namespace services
{

/**
 * @brief: ST 2110-20 SRD RTP extension header wire format.
 *
 * RTP header extension for ST 2110-20 Sample Row Data (SRD) packets per RFC 4175
 * section 4.2. Shared by packet readers and writers for parsing/encoding the SRD block.
 *
 * 0                   1                   2                   3
 * 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |    Extended Sequence Number   |           SRD Length          |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |F|     SRD Row Number          |C|         SRD Offset          |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 */
struct SRDHeader {
    uint16_t sequence_number;                 /**< Extended Sequence Number high 16 bits (network byte order on wire) */
    uint16_t srd_length;                      /**< SRD Length (network byte order on wire) */

    uint8_t srd_line_number_8_to_14_7bit : 7; /**< SRD line number high 7 bits */
    uint8_t f : 1;                            /**< Field indicator (interlaced) */
    uint8_t srd_line_number_0_to_7_8bit;      /**< SRD line number low 8 bits */

    uint8_t srd_offset_8_to_14_7bit : 7;      /**< SRD offset high 7 bits */
    uint8_t c : 1;                            /**< Field identification */
    uint8_t srd_offset_0_to_7_8bit;           /**< SRD offset low 8 bits */

    /** 
     * @brief: Sets the SRD line number.
     *
     * @param [in] srd_line_number: The SRD line number to set.
     */
    void set_srd_line_number(uint16_t srd_line_number)
    {
        srd_line_number_0_to_7_8bit = static_cast<uint8_t>(srd_line_number);
        srd_line_number_8_to_14_7bit = static_cast<uint8_t>(srd_line_number >> 8);
    }
    /**
     * @brief: Sets the SRD offset.
     *
     * @param [in] srd_offset: The SRD offset to set.
     */
    void set_srd_offset(uint16_t srd_offset)
    {
        srd_offset_0_to_7_8bit = static_cast<uint8_t>(srd_offset);
        srd_offset_8_to_14_7bit = static_cast<uint8_t>(srd_offset >> 8);
    }
    /**
     * @brief: Returns the SRD line number.
     *
     * @return: The SRD line number.
     */
    uint16_t get_srd_line_number() const
    {
        return static_cast<uint16_t>((srd_line_number_8_to_14_7bit << 8) | srd_line_number_0_to_7_8bit);
    }
    /**
     * @brief: Returns the SRD offset.
     *
     * @return: The SRD offset.
     */
    uint16_t get_srd_offset() const
    {
        return static_cast<uint16_t>((srd_offset_8_to_14_7bit << 8) | srd_offset_0_to_7_8bit);
    }
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_ULP_PACKET_RTP_SMPTE_2110_20_EXTENSION_H_ */
