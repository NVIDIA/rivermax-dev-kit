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

#ifndef RDK_EXAMPLES_APPS_MEDIA_SENDER_NODE_SHUTDOWN_VIDEO_FILE_SENDER_H_
#define RDK_EXAMPLES_APPS_MEDIA_SENDER_NODE_SHUTDOWN_VIDEO_FILE_SENDER_H_

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
/**
 * @brief: Node shutdown video file sender example.
 *
 * This example sends a video file the same way @ref UserProvidedSettingsVideoFileSenderExample
 * does, over two I/O nodes rather than one, and shuts them down through the node and
 * application lifecycle API: taking a node id from the settings the application exposes,
 * stopping that node alone, waiting for it, and then stopping the whole application and
 * waiting for it.
 */
class NodeShutdownVideoFileSenderExample : public BaseExample
{
public:
    /**
     * @brief: Constructor for the @ref NodeShutdownVideoFileSenderExample class.
     */
    NodeShutdownVideoFileSenderExample();

protected:
    ReturnStatus operator()() override;
    void add_cli_options() override;

private:
    /**
     * @brief: Configures the example-specific media sender settings.
     *
     * @param [out] settings: Reference to @ref MediaSenderSettings to be configured.
     */
    void configure_media_sender_settings(MediaSenderSettings& settings) const;
};

} // namespace apps
} // namespace examples
} // namespace rdk

#endif /* RDK_EXAMPLES_APPS_MEDIA_SENDER_NODE_SHUTDOWN_VIDEO_FILE_SENDER_H_ */
