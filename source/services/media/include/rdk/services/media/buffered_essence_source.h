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

#ifndef RDK_SERVICES_MEDIA_BUFFERED_ESSENCE_SOURCE_H_
#define RDK_SERVICES_MEDIA_BUFFERED_ESSENCE_SOURCE_H_

#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>

#include "rdk/services/media/media_essence_source.h"
#include "rdk/services/media/media_settings.h"
#include "rdk/services/memory_allocation/memory_allocation.h"
#include "rdk/services/utils/defs.h"

namespace rdk
{
namespace services
{
/**
 * @brief: Provide media units from a buffer queue, implementing @ref IMediaEssenceSource.
 *
 * This class maintains a queue of media units and provides them to consumers.
 * Units can be added to the queue by producers and will be automatically
 * returned to their source when no longer needed.
 */
class BufferedEssenceSource : public IMediaEssenceSource
{
private:
    /* Queue of media units */
    std::queue<std::shared_ptr<MediaUnit>> m_media_unit_queue;
    /* Mutex for thread safety */
    mutable std::mutex m_mutex;
    /* Condition variable for blocking operations */
    std::condition_variable m_cv;
    /* Maximum size of the queue (0 for unlimited) */
    size_t m_max_queue_size;
    /* Flag to indicate if the source is stopping */
    bool m_stop;
public:
    /**
     * @brief: Constructor.
     *
     * @param [in] max_queue_size: Maximum size of the internal media unit queue (0 for unlimited).
     */
    BufferedEssenceSource(size_t max_queue_size = 0);
    /**
     * @brief: Destructor.
     */
    virtual ~BufferedEssenceSource();
    std::shared_ptr<MediaUnit> get_media_unit_blocking() override;
    std::shared_ptr<MediaUnit> get_media_unit_non_blocking() override;
    void stop() override;
    /**
     * @brief: Add a media unit to the queue.
     *
     * @param [in] media_unit: Shared pointer to a @ref MediaUnit.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus add_media_unit(std::shared_ptr<MediaUnit> media_unit);
    /**
     * @brief: Return the number of media units in the queue.
     *
     * @return: Number of media units in the queue.
     */
    size_t get_queue_size() const;
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_MEDIA_BUFFERED_ESSENCE_SOURCE_H_ */
