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

#ifndef RDK_EXAMPLES_APPS_RDK_MEDIA_SENDER_USER_PROVIDED_SETTINGS_VIDEO_FILE_SENDER_H_
#define RDK_EXAMPLES_APPS_RDK_MEDIA_SENDER_USER_PROVIDED_SETTINGS_VIDEO_FILE_SENDER_H_

#include "rdk/apps/media_sender/media_sender.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/examples/base_example.h"

using namespace rivermax::dev_kit::apps::media_sender;
using namespace rivermax::dev_kit::services;

namespace rivermax
{
namespace dev_kit
{
namespace examples
{
namespace apps
{
namespace rdk_media_sender
{
/**
 * @brief: User-provided settings video file sender example.
 *
 * This example demonstrates sending video files using user-provided application settings from
 * the app level module.
 */
class UserProvidedSettingsVideoFileSenderExample : public BaseExample
{
public:
    /**
     * @brief: Constructor for the @ref UserProvidedSettingsVideoFileSenderExample class.
     */
    UserProvidedSettingsVideoFileSenderExample();

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

} // namespace rdk_media_sender
} // namespace apps
} // namespace examples
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_EXAMPLES_APPS_RDK_MEDIA_SENDER_USER_PROVIDED_SETTINGS_VIDEO_FILE_SENDER_H_ */
