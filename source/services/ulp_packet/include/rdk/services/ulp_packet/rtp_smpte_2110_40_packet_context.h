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

#ifndef RDK_SERVICES_ULP_PACKET_RTP_SMPTE_2110_40_PACKET_CONTEXT_H_
#define RDK_SERVICES_ULP_PACKET_RTP_SMPTE_2110_40_PACKET_CONTEXT_H_

#include "rdk/services/ulp_packet/rtp_packet_context.h"
#include "rdk/services/media/ancillary_metadata.h"

namespace rdk
{
namespace services
{

constexpr uint8_t RTP_2110_40_FIELD_INDICATOR_PROGRESSIVE = 0;
constexpr uint8_t RTP_2110_40_FIELD_INDICATOR_FIELD1 = 2;
constexpr uint8_t RTP_2110_40_FIELD_INDICATOR_FIELD2 = 3;

/**
 * @brief: Context for RTP SMPTE 2110-40 packets.
 *
 * This struct extends the @ref RTPPacketContext to include fields
 * specific to SMPTE 2110-40 ancillary data packets based on section 2.1 of
 * RFC 8331 - RTP Payload for SMPTE ST 291-1 Ancillary Data.
 */
struct RTP_SMPTE_2110_40_PacketContext : public RTPPacketContext
{
    uint32_t extended_sequence_number = 0;                              /**< 32-bit extended sequence number */
    uint16_t length = 0;                                                /**< Number of octets of the ANC data RTP payload */
    uint32_t ancillary_count = 0;                                       /**< Number of ancillary data packets */
    uint8_t field_indicator = RTP_2110_40_FIELD_INDICATOR_PROGRESSIVE;  /**< Field indicator */
    size_t descriptor_start_index = 0;                                  /**< Index of first descriptor in this RTP packet */
    size_t descriptor_count_in_packet = 0;                              /**< Number of descriptors in this RTP packet */

    /** Pointer to ancillary descriptors: writer reads from it, reader appends to it */
    std::vector<AncillaryDataDescriptor>* descriptors = nullptr;
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_ULP_PACKET_RTP_SMPTE_2110_40_PACKET_CONTEXT_H_ */
