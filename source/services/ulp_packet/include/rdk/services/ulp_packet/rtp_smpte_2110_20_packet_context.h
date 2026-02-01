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

#ifndef RDK_SERVICES_ULP_PACKET_RTP_SMPTE_2110_20_PACKET_CONTEXT_H_
#define RDK_SERVICES_ULP_PACKET_RTP_SMPTE_2110_20_PACKET_CONTEXT_H_

#include "rdk/services/ulp_packet/rtp_packet_context.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Context for RTP SMPTE 2110-20 packets.
 *
 * This struct extends the @ref RTPPacketContext to include fields
 * specific to SMPTE 2110-20 video frames packets based on section 4.2 of
 * RFC 4175 - RTP Payload Format for Uncompressed Video.
 */
struct RTP_SMPTE_2110_20_PacketContext : public RTPPacketContext
{
    uint32_t extended_sequence_number = 0;          /**< 32-bit extended sequence number */
    uint16_t srd_length = 0;                        /**< Number of octets of data included from this scan line */
    uint16_t line_number = 0;                       /**< Scan line number of encapsulated data */
    uint16_t srd_offset = 0;                        /**< Offset of the first pixel of the payload data within the scan line */
    uint8_t rtp_interlace_field_indicator = 0;      /**< Identifies which field the scan line belongs to,
                                                         for interlaced data */
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_ULP_PACKET_RTP_SMPTE_2110_20_PACKET_CONTEXT_H_ */
