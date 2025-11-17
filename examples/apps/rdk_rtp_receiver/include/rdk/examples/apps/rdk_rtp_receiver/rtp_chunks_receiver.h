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

#ifndef RDK_EXAMPLES_APPS_RDK_RTP_RECEIVER_RTP_CHUNKS_RECEIVER_H_
#define RDK_EXAMPLES_APPS_RDK_RTP_RECEIVER_RTP_CHUNKS_RECEIVER_H_

#include "rdk/apps/rtp_receiver/rtp_receiver.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/examples/base_example.h"

using namespace rivermax::dev_kit::apps::rtp_receiver;
using namespace rivermax::dev_kit::services;

namespace rivermax
{
namespace dev_kit
{
namespace examples
{
namespace apps
{
namespace rdk_rtp_receiver
{

/**
 * @brief: RTP chunks receiver example.
 *
 * This example demonstrates consuming chunks received over the network from the app level module.
 * It uses a dummy data consumer that simulates data consumption from the RTP receiver.
 */
class RTPChunksReceiverExample : public BaseExample
{
public:
    /**
     * @brief: Constructor for the @ref RTPChunksReceiverExample class.
     */
    RTPChunksReceiverExample();

protected:
    ReturnStatus operator()() override;
    void add_cli_options() override;

private:
    /**
     * @brief: Configures the example-specific RTP receiver settings.
     *
     * @param [out] settings: Reference to @ref RTPReceiverSettings to be configured.
     */
    void configure_rtp_receiver_settings(RTPReceiverSettings& settings) const;
};

} // namespace rdk_rtp_receiver
} // namespace apps
} // namespace examples
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_EXAMPLES_APPS_RDK_RTP_RECEIVER_RTP_CHUNKS_RECEIVER_H_ */
