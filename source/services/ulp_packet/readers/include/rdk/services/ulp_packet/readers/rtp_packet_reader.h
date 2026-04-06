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

#ifndef RDK_SERVICES_ULP_PACKET_READERS_RTP_PACKET_READER_H_
#define RDK_SERVICES_ULP_PACKET_READERS_RTP_PACKET_READER_H_

#include "rdk/services/ulp_packet/readers/ulp_packet_reader_interface.h"
#include "rdk/services/ulp_packet/rtp_header.h"
#include "rdk/services/ulp_packet/rtp_packet_context.h"

namespace rdk
{
namespace services
{

/**
 * @brief: Base class for RTP media packet readers.
 *
 * This class provides methods to parse the RTP header and payload for RTP media packets.
 */
class RTPPacketReader : public IULPPacketReader {
public:
    RTPPacketReader(const byte_t* header_ptr, const byte_t* payload_ptr = nullptr)
        : IULPPacketReader(header_ptr, payload_ptr) {}
    virtual ~RTPPacketReader() = default;

    ReturnStatus parse_header(IPacketContext& context) override;
    inline size_t get_payload_size() const override { return m_payload_size; }
    inline size_t get_header_size() const override { return sizeof(RTPHeader); }

protected:
    size_t m_payload_size = 0;
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_ULP_PACKET_READERS_RTP_PACKET_READER_H_ */
