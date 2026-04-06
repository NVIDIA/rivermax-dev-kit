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

#ifndef RDK_SERVICES_PROTOCOL_RTP_H_
#define RDK_SERVICES_PROTOCOL_RTP_H_

#include <cstdint>

namespace rdk
{
namespace services
{

namespace protocol
{

#pragma pack(push, 1)

/**
 * @brief: RTP fixed header (RFC 3550 section 5.1).
 *
 * Represents the mandatory 12-byte portion that precedes any CSRC list or
 * extension headers. All multi-byte integers are stored in <b>network byte
 * order</b> (big-endian).
 */
struct RTPHeader
{
    /** Version/Padding/Extension/CSRC count:
     *  - <b>bits 7-6</b>: Version (must be 2)
     *  - <b>bit 5</b>: Padding flag
     *  - <b>bit 4</b>: Header-extension flag
     *  - <b>bits 3-0</b>: CSRC count */
    uint8_t vpxcc;
    /** Marker / Payload-type:
     *  - <b>bit 7</b>: Marker flag (M)
     *  - <b>bits 6-0</b>: Payload-type (PT) */
    uint8_t mpt;
    /** Sequence number. */
    uint16_t sequence_number;
    /** RTP timestamp. */
    uint32_t timestamp;
    /** Synchronization source identifier (SSRC). */
    uint32_t ssrc;
};

#pragma pack(pop)

} // namespace protocol

} // namespace services
} // namespace rdk

#endif // RDK_SERVICES_PROTOCOL_RTP_H_