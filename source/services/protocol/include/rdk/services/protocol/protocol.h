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

#ifndef RDK_SERVICES_PROTOCOL_PROTOCOL_H_
#define RDK_SERVICES_PROTOCOL_PROTOCOL_H_

// Protocol header definitions
#include "rdk/services/protocol/ethernet.h"
#include "rdk/services/protocol/ipv4.h"
#include "rdk/services/protocol/udp.h"
#include "rdk/services/protocol/rtp.h"

// Protocol parsers
#include "rdk/services/protocol/protocol_parsers.h"
#include "rdk/services/protocol/ethernet_parser.h"
#include "rdk/services/protocol/ipv4_parser.h"
#include "rdk/services/protocol/udp_parser.h"
#include "rdk/services/protocol/rtp_parser.h"

// High-level packet parsers
#include "rdk/services/protocol/packet_parser.h"
#include "rdk/services/protocol/media_packet_parser.h"

#endif // RDK_SERVICES_PROTOCOL_PROTOCOL_H_
