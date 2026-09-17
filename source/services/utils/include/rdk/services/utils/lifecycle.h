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

#ifndef RDK_SERVICES_UTILS_LIFECYCLE_H_
#define RDK_SERVICES_UTILS_LIFECYCLE_H_

#include <chrono>

#include "rdk/services/error_handling/return_status.h"

namespace rdk
{
namespace services
{

/**
 * @brief: Interface for a component with a start, a stop and a completion to wait for.
 *
 * Only @ref stop is pure. @ref start and @ref await default to a no-op, for a passive
 * component that runs no thread of its own.
 *
 * @note: A component that owns other @ref ILifecycle objects must call @ref stop on each of
 *        them from its own @ref stop, except one shared with siblings, which must use that
 *        object's own protocol for departing instead.
 */
class ILifecycle
{
public:
    virtual ~ILifecycle() = default;
    /**
     * @brief: Starts the component's work.
     *
     * Blocks until the work finishes.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus start() { return ReturnStatus::success; }
    /**
     * @brief: Requests the component to stop.
     *
     * Non-blocking, idempotent, thread-safe. Does not wait for the work to finish.
     */
    virtual void stop() noexcept = 0;
    /**
     * @brief: Waits for the component's work to finish.
     *
     * Checks once; if not finished, sleeps for the full @p timeout, then checks once more.
     * Call again to keep waiting.
     *
     * @param [in] timeout: Maximum time to wait.
     *
     * @return: @ref ReturnStatus::success if the work finished; @ref ReturnStatus::still_active
     *          if @p timeout elapsed first.
     */
    virtual ReturnStatus await(std::chrono::nanoseconds) const { return ReturnStatus::success; }
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_UTILS_LIFECYCLE_H_ */
