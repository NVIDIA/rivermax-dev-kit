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

#ifndef RDK_EXAMPLES_APPS_INTEGRATION_MEDIA_RECEIVER_SENDER_H_
#define RDK_EXAMPLES_APPS_INTEGRATION_MEDIA_RECEIVER_SENDER_H_

#include "rdk/apps/rtp_receiver/rtp_receiver.h"
#include "rdk/apps/media_sender/media_sender.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/examples/base_example.h"

using namespace rdk::apps;
using namespace rdk::services;

namespace rdk
{
namespace examples
{
namespace apps
{
namespace integration
{

/**
 * @brief: Combined receiver and sender example.
 *
 * This example demonstrates both receiving and sending data in a single application.
 * It uses the default dummy data consumer to simulate processing of chunks received over the network
 * by the RTP receiver. Additionally, it uses the default dummy frame source to simulate frames transmission
 * from the sender. See @ref RTPChunksReceiverExample and @ref VideoFramesSenderExample for details about
 * overriding the default dummy data consumer and frame source.
 * The example is designed to showcase the integration of both receiver and sender functionalities.
 */
class ReceiverSenderExample : public BaseExample
{
public:
    /**
     * @brief: Constructor for the @ref ReceiverSenderExample class.
     */
    ReceiverSenderExample();

protected:
    ReturnStatus operator()() override;
    void add_cli_options() override;

private:
    /**
     * @brief: Configures the example-specific RTP receiver and media sender settings.
     *
     * @param [out] rtp_receiver_settings: Reference to @ref RTPReceiverSettings to be configured.
     * @param [out] media_sender_settings: Reference to @ref MediaSenderSettings to be configured.
     */
    void configure_rtp_receiver_media_sender_settings(
        RTPReceiverSettings& rtp_receiver_settings,
        MediaSenderSettings& media_sender_settings) const;
};

} // namespace integration
} // namespace apps
} // namespace examples
} // namespace rdk

#endif /* RDK_EXAMPLES_APPS_INTEGRATION_MEDIA_RECEIVER_SENDER_H_ */
