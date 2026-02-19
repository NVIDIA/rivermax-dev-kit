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

#ifndef RDK_SERVICES_ULP_PACKET_READERS_RTP_SMPTE_2110_40_PACKET_READER_H_
#define RDK_SERVICES_ULP_PACKET_READERS_RTP_SMPTE_2110_40_PACKET_READER_H_

#include "rdk/services/ulp_packet/readers/rtp_packet_reader.h"
#include "rdk/services/ulp_packet/rtp_smpte_2110_40_packet_context.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: RTP packet reader for SMPTE 2110-40 ancillary data.
 *
 * This class provides methods to parse the RTP header and payload for SMPTE
 * 2110-40 ancillary data packets based on RFC 8331 - RTP Payload for SMPTE 
 * ST 291-1 Ancillary Data.
 */
class RTP_SMPTE_2110_40_PacketReader : public RTPPacketReader
{
public:
    RTP_SMPTE_2110_40_PacketReader(const byte_t* header_ptr, const byte_t* payload_ptr = nullptr)
        : RTPPacketReader(header_ptr, payload_ptr) {}
    virtual ~RTP_SMPTE_2110_40_PacketReader() = default;

    ReturnStatus parse_header(IPacketContext& context) override;
    ReturnStatus parse_payload(IPacketContext& context) override;
    inline size_t get_header_size() const override { return 0; /* TODO: Implement actual header size calculation */ }
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_ULP_PACKET_READERS_RTP_SMPTE_2110_40_PACKET_READER_H_ */
