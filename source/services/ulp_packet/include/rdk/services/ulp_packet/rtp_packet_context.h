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

#ifndef RDK_SERVICES_ULP_PACKET_RTP_PACKET_CONTEXT_H_
#define RDK_SERVICES_ULP_PACKET_RTP_PACKET_CONTEXT_H_

#include "rdk/services/ulp_packet/ulp_packet_context.h"
#include "rdk/services/utils/defs.h"
#include "rdk/services/utils/rational.h"

namespace rdk
{
namespace services
{

/**
 * @brief: Context for RTP packets.
 *
 * This struct extends @ref IPacketContext with RTP-specific fields based on Section 5.1
 * of RFC 3550 - RTP: A Transport Protocol for Real-Time Applications.
 */
struct RTPPacketContext : public IPacketContext
{
    uint8_t version = 2;                 /**< RTP version, currently 2 */
    bool padding = false;                /**< Padding bit */
    bool extension = false;              /**< Extension bit */
    uint8_t cc = 0;                      /**< CSRC count */
    uint8_t payload_type = 0;            /**< Payload type */
    bool marker = false;                 /**< Marker bit */
    uint16_t sequence = 0;               /**< 16-bit RTP sequence number */
    Rational timestamp;                  /**< 32-bit RTP timestamp (Rational for arithmetic precision) */
    uint32_t ssrc = 0;                   /**< Synchronization source (SSRC) identifier */

    byte_t* payload_ptr = nullptr;       /**< Pointer to payload for this packet */
    size_t payload_size = 0;             /**< Size of the payload for this packet */
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_ULP_PACKET_RTP_PACKET_CONTEXT_H_ */
