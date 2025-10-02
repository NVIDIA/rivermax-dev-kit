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

#ifndef RDK_SERVICES_MEDIA_NULL_ESSENCE_PROVIDER_H_
#define RDK_SERVICES_MEDIA_NULL_ESSENCE_PROVIDER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <unordered_map>

#include "rdk/services/media/media_essence_provider.h"
#include "rdk/services/utils/defs.h"
#include "rdk/services/media/media_settings.h"
#include "rdk/services/memory_allocation/memory_allocation.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 * @brief: Mock implementation of IMediaEssenceProvider for testing.
 */
class NullEssenceProvider : public IMediaEssenceProvider {
protected:
    const MediaSettings& m_media_settings;
    MediaUnitMetadata m_cached_metadata;
    size_t m_cached_media_unit_size;
    float m_media_unit_not_available_probability;
public:
    /**
     * @brief: Constructor of NullEssenceProvider.
     *
     * @param [in] media_settings: Media settings for the generated stream.
     */
    NullEssenceProvider(const MediaSettings& media_settings);

    std::shared_ptr<MediaUnit> get_media_unit_blocking() override;
    std::shared_ptr<MediaUnit> get_media_unit_non_blocking() override;
    /**
     * @brief: Sets the probability of returning nullptr in @ref get_media_unit_non_blocking.
     *
     * @param [in] probability: Probability value between 0 and 1.
     *
     * @return: Status of the operation.
     */
    ReturnStatus set_media_unit_not_available_probability(float probability);
    /**
     * @brief: Returns the current probability of returning nullptr in @ref get_media_unit_non_blocking.
     *
     * @return: Probability value between 0 and 1.
     */
    float get_media_unit_not_available_probability() const  { return m_media_unit_not_available_probability; }
private:
    /**
     * @brief: Sets media unit settings based on the stream type.
     *
     * @param [in] metadata: Metadata to set.
     * @param [in] media_unit_size: Size of the media unit to set.
     */
    void set_media_unit_settings(MediaUnitMetadata& metadata, size_t& media_unit_size);
    /**
     * @brief: Determines if a media unit should be available based on the probability.
     *
     * @return: True if the media unit should be available, false otherwise.
     */
    bool is_media_unit_available() const;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_NULL_ESSENCE_PROVIDER_H_ */
