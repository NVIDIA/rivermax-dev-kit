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

#ifndef RDK_IO_NODE_IO_NODE_SETTINGS_H_
#define RDK_IO_NODE_IO_NODE_SETTINGS_H_

#include <cstddef>

#include "rdk/services/error_handling/return_status.h"

using namespace rdk::services;

namespace rdk
{
namespace io_node
{

/**
 * @brief: Base settings common to every I/O node.
 */
struct IONodeSettings
{
    IONodeSettings() = default;
    virtual ~IONodeSettings() = default;

    /* Position of this node in the app's node container. */
    size_t id = 0;
    /**
     * @brief: Checks the settings a node cannot be built without.
     *
     * Construction-time requirements only, not fields an application fills in later.
     * An override calls its base first.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus validate() const
    {
        return ReturnStatus::success;
    }
};

} // namespace io_node
} // namespace rdk

#endif /* RDK_IO_NODE_IO_NODE_SETTINGS_H_ */
