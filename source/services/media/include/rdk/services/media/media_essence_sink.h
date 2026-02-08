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

#ifndef RDK_SERVICES_MEDIA_MEDIA_ESSENCE_SINK_H_
#define RDK_SERVICES_MEDIA_MEDIA_ESSENCE_SINK_H_

#include <memory>

#include "rdk/services/media/media_essence_source.h"
#include "rdk/services/utils/defs.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 * @brief: Interface for media essence sinks.
 */
class IMediaEssenceSink
{
public:
    virtual ~IMediaEssenceSink() = default;
    /**
     * @brief: Receives a media unit in a blocking manner.
     *
     * This method blocks if the sink cannot accept the media unit immediately
     * (e.g., internal queue is full). Returns when the media unit is accepted
     * or the sink is stopped.
     *
     * @param [in] media_unit: Shared pointer to the media unit to receive.
     *
     * @return: @ref ReturnStatus::success when accepted, @ref ReturnStatus::failure on error.
     */
    virtual ReturnStatus put_media_unit_blocking(std::shared_ptr<MediaUnit> media_unit) = 0;
    /**
     * @brief: Receives a media unit in a non-blocking manner.
     *
     * Returns immediately. If the sink is not ready to accept (e.g. internal queue full),
     * returns @ref ReturnStatus::sink_not_ready.
     *
     * @param [in] media_unit: Shared pointer to the media unit to receive.
     *
     * @return: @ref ReturnStatus::success if accepted, @ref ReturnStatus::sink_not_ready if the sink
     *          is not ready to accept, @ref ReturnStatus::failure on error.
     */
    virtual ReturnStatus put_media_unit_non_blocking(std::shared_ptr<MediaUnit> media_unit) = 0;
    /**
     * @brief: Stop the sink and release all waiting threads.
     */
    virtual void stop() {}

protected:
    IMediaEssenceSink() = default;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_MEDIA_ESSENCE_SINK_H_ */
