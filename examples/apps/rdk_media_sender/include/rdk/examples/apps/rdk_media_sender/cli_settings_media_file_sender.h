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

#ifndef RDK_EXAMPLES_APPS_RDK_MEDIA_SENDER_CLI_SETTINGS_MEDIA_FILE_SENDER_H_
#define RDK_EXAMPLES_APPS_RDK_MEDIA_SENDER_CLI_SETTINGS_MEDIA_FILE_SENDER_H_

#include "rdk/services/error_handling/return_status.h"
#include "rdk/examples/base_example.h"

using namespace rdk::services;

namespace rdk
{
namespace examples
{
namespace apps
{
/**
 * @brief: CLI settings media file sender example.
 *
 * This example demonstrates sending media file using CLI settings from
 * the app level module.
 */
class CLISettingsMediaFileSenderExample : public BaseExample
{
public:
    /**
     * @brief: Constructor for the @ref CLISettingsMediaFileSenderExample class.
     */
    CLISettingsMediaFileSenderExample();

    int run(int argc, const char* argv[]) override;

protected:
    ReturnStatus operator()() override;

private:
    /** Stored command line arguments count */
    int m_argc = 0;
    /** Stored command line arguments */
    const char** m_argv = nullptr;
};

} // namespace apps
} // namespace examples
} // namespace rdk

#endif /* RDK_EXAMPLES_APPS_RDK_MEDIA_SENDER_CLI_SETTINGS_MEDIA_FILE_SENDER_H_ */
