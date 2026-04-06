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

#ifndef RDK_SERVICES_ULP_PACKET_RTP_SMPTE_2110_40_EXTENSION_H_
#define RDK_SERVICES_ULP_PACKET_RTP_SMPTE_2110_40_EXTENSION_H_

#include <cstdint>

namespace rdk
{
namespace services
{

/**
 * @brief: ST 2110-40 Ancillary RTP extension header.
 *
 * RTP header extension for ancillary data streams per RFC 8331 section 2.1.
 *
 * 0                   1                   2                   3
 * 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |   Extended Sequence Number    |           Length              |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * | ANC_Count     | F |                reserved                   |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 */
struct AncillaryRTPExtension
{
    uint16_t extended_sequence_number;  /**< High 16 bits of extended 32-bit sequence number (network byte order on wire) */
    uint16_t length;                    /**< Octets of ANC data RTP payload from first ANC packet header (network byte order on wire) */
    uint8_t anc_count;                  /**< Count of ANC data packets in this RTP payload (0-255) */
    uint8_t reserved_byte1;             /**< F in bits 6-7: 0b00 progressive, 0b10 field1, 0b11 field2; bits 0-5 reserved */
    uint16_t reserved_bytes2_3;         /**< Reserved (22 bits total with bits 0-5 of reserved_byte1) */

    /**
     * @brief: Set the field indicator (0=progressive, 2=field1, 3=field2).
     *
     * @param [in] f: The field indicator value to set.
     */
    void set_field_indicator(uint8_t f)
    {
        reserved_byte1 = (reserved_byte1 & 0x3F) | ((f & 0x03) << 6);
    }
    /**
     * @brief: Get the field indicator (0=progressive, 2=field1, 3=field2).
     *
     * @return: The field indicator value.
     */
    uint8_t get_field_indicator() const
    {
        return (reserved_byte1 >> 6) & 0x03;
    }
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_ULP_PACKET_RTP_SMPTE_2110_40_EXTENSION_H_ */
