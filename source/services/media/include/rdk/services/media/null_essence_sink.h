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

#ifndef RDK_SERVICES_MEDIA_NULL_ESSENCE_SINK_H_
#define RDK_SERVICES_MEDIA_NULL_ESSENCE_SINK_H_

#include <memory>

#include "rdk/services/media/media_essence_sink.h"
#include "rdk/services/utils/defs.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 * @brief: Null implementation of @ref IMediaEssenceSink for testing and default behavior.
 *
 * Discards all received media units. Used as default sink when no external sink
 * is configured.
 */
class NullEssenceSink : public IMediaEssenceSink
{
public:
    /**
     * @brief: Constructor for NullEssenceSink.
     */
    NullEssenceSink() : m_put_failure_probability(0.0f) {}
    /**
     * @brief: Receives and discards a media unit (blocking).
     *
     * @param [in] media_unit: Shared pointer to the media unit to receive.
     *
     * @return: @ref ReturnStatus::success when discarded, @ref ReturnStatus::failure on error.
     */
    ReturnStatus put_media_unit_blocking(std::shared_ptr<MediaUnit> media_unit) override;
    /**
     * @brief: Receives and discards a media unit (non-blocking).
     *
     * @param [in] media_unit: Shared pointer to the media unit to receive.
     *
     * @return: @ref ReturnStatus::success if accepted, @ref ReturnStatus::sink_not_ready when simulated
     *          backpressure is triggered, @ref ReturnStatus::failure on error.
     */
    ReturnStatus put_media_unit_non_blocking(std::shared_ptr<MediaUnit> media_unit) override;
    /**
     * @brief: Sets the probability of returning @ref ReturnStatus::sink_not_ready in @ref put_media_unit_non_blocking.
     *
     * Used for testing backpressure handling.
     *
     * @param [in] probability: Probability value between 0 and 1.
     *
     * @return: Status of the operation.
     */
    ReturnStatus set_put_failure_probability(float probability);
    /**
     * @brief: Returns the current probability of returning @ref ReturnStatus::sink_not_ready.
     *
     * @return: Probability value between 0 and 1.
     */
    float get_put_failure_probability() const { return m_put_failure_probability; }

private:
    float m_put_failure_probability;
    /**
     * @brief: Determines if the sink can accept a media unit.
     *
     * @return: True if the sink can accept (put will succeed), false otherwise.
     */
    bool can_accept_media_unit() const;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_NULL_ESSENCE_SINK_H_ */
