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

#include "rdk/services/ulp_packet/readers/rtp_smpte_2110_40_packet_reader.h"

using namespace rivermax::dev_kit::services;

ReturnStatus RTP_SMPTE_2110_40_PacketReader::parse_header(IPacketContext& context)
{
    // TODO: Implement ST 2110-40 header parsing
    return ReturnStatus::success;
}

ReturnStatus RTP_SMPTE_2110_40_PacketReader::parse_payload(IPacketContext& context)
{
    // TODO: Implement ST 2110-40 payload parsing
    return ReturnStatus::success;
}
