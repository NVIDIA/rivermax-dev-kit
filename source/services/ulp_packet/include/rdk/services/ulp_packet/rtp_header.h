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

#ifndef RDK_SERVICES_ULP_PACKET_RTP_HEADER_H_
#define RDK_SERVICES_ULP_PACKET_RTP_HEADER_H_

#include <cstdint>

namespace rdk
{
namespace services
{

/**
 * @brief: Standard RTP packet header wire format.
 *
 * Layout matches RFC 3550 - RTP: A Transport Protocol for Real-Time Applications.
 * Shared by packet readers and writers for parsing/encoding the base RTP header.
 */
struct RTPHeader {
    uint8_t cc : 4;             /**< CSRC count */
    uint8_t extension : 1;      /**< Extension bit */
    uint8_t padding : 1;        /**< Padding bit */
    uint8_t version : 2;        /**< RTP version */
    uint8_t payload_type : 7;   /**< Payload type */
    uint8_t marker : 1;         /**< Marker bit */
    uint16_t sequence_number;   /**< Sequence number (network byte order on wire) */
    uint32_t timestamp;         /**< Timestamp (network byte order on wire) */
    uint32_t ssrc;              /**< Synchronization source (SSRC) (network byte order on wire) */
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_ULP_PACKET_RTP_HEADER_H_ */
