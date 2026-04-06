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

#ifndef RDK_EXAMPLES_SERVICES_ULP_PACKET_BUFFER_WRITERS_SMPTE_2110_20_RTP_BUFFER_WRITER_H_
#define RDK_EXAMPLES_SERVICES_ULP_PACKET_BUFFER_WRITERS_SMPTE_2110_20_RTP_BUFFER_WRITER_H_

#include <memory>

#include "rdk/services/settings/app_settings.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/examples/base_example.h"

using namespace rdk::services;

namespace rdk
{
namespace examples
{
namespace services
{

/**
 * @brief: SMPTE 2110-20 RTP packets buffer writer example.
 *
 * This example demonstrates the writing of RTP packets to a buffer for SMPTE 2110-20 streams,
 * using @ref RTPMediaPacketBufferWriter<RTP_SMPTE_2110_20_PacketContext, RTP_SMPTE_2110_20_PacketWriter>
 */
class RTP_SMPTE_2110_20_PacketBufferWriterExample : public BaseExample
{
public:
    /**
     * @brief: Constructor for the @ref RTP_SMPTE_2110_20_PacketBufferWriterExample class.
     */
    RTP_SMPTE_2110_20_PacketBufferWriterExample();

    ReturnStatus operator()() override;
    void add_cli_options() override {};

private:
    /**
     * @brief: Configure application settings with example-specific values.
     *
     * @param [out] settings: Reference to @ref AppSettings to be configured.
     */
    void configure_app_settings(AppSettings& settings) const;
};

} // namespace services
} // namespace examples
} // namespace rdk

#endif /* RDK_EXAMPLES_SERVICES_ULP_PACKET_BUFFER_WRITERS_SMPTE_2110_20_RTP_BUFFER_WRITER_H_ */
