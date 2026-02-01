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

#ifndef RDK_SERVICES_MEDIA_MEDIA_UNIT_POOL_H_
#define RDK_SERVICES_MEDIA_MEDIA_UNIT_POOL_H_

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include "rdk/services/media/media_essence_source.h"
#include "rdk/services/memory_allocation/memory_allocation.h"
#include "rdk/services/utils/defs.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Manages a pool of pre-allocated @ref MediaUnit objects.
 *
 * This class allocates a contiguous block of memory and divides it into
 * fixed-size media units. Media units can be borrowed from the pool and are automatically
 * returned when no longer in use. The pool ensures efficient memory usage and
 * provides thread-safe access to the media units.
 *
 * The @ref MediaUnitPool class supports both non-blocking and blocking methods for
 * obtaining media units. The non-blocking method returns a media unit if available, or
 * nullptr if no media units are available. The blocking method waits until a media unit
 * becomes available.
 *
 * The class also provides methods to query the number of available units and
 * the total number of units in the pool.
 */
class MediaUnitPool {
private:
    /* The total memory block that contains all media units */
    byte_t* m_memory_block;
    /* Vector of all media units for tracking */
    std::vector<MediaUnit> m_all_media_units;
    /* Queue of available media units */
    std::queue<size_t> m_available_indices;
    /* Mutex for thread safety */
    mutable std::mutex m_mutex;
    /* Condition variable for blocking operations */
    std::condition_variable m_cv;
    /* Size of each media unit */
    size_t m_media_unit_size;
    /* Total number of media units */
    size_t m_media_unit_count;
    /* Total allocated memory size */
    size_t m_total_memory_size;
    /* Flag that signals that object destruction has been started */
    std::atomic<bool> m_in_destruction{false};
    /* Flag to indicate if the pool is stopping */
    std::atomic<bool> m_stop{false};
    /* Memory location used for allocation */
    MemoryLocation m_memory_location;
    /* SMPTE standard for media units */
    SMPTEStandard m_smpte_standard;
public:
    /**
     * @brief: Constructor.
     *
     * @param [in] media_unit_count: Number of media units to allocate in the pool.
     * @param [in] media_unit_size: Size of each media unit in bytes.
     * @param [in] smpte_standard: SMPTE standard for media units.
     * @param [in] mem_allocator: Memory allocator to use.
     */
    MediaUnitPool(size_t media_unit_count, size_t media_unit_size, SMPTEStandard smpte_standard,
        MemoryAllocator& mem_allocator);
    /**
     * @brief: Constructor for external memory.
     *
     * @param [in] media_unit_count: Number of media units in the pool.
     * @param [in] media_unit_size: Size of each media unit in bytes.
     * @param [in] smpte_standard: SMPTE standard for media units.
     * @param [in] memory_block: Pointer to the external memory block.
     * @param [in] memory_size: Size of the external memory block in bytes.
     * @param [in] memory_location: Memory location of the external memory block.
     *
     * @throws std::invalid_argument if the provided memory is insufficient.
     */
    MediaUnitPool(size_t media_unit_count, size_t media_unit_size, SMPTEStandard smpte_standard,
        byte_t* memory_block, size_t memory_size,
        MemoryLocation memory_location = MemoryLocation::Host);
    /**
     * @brief: Destructor.
     */
    virtual ~MediaUnitPool();
    /**
     * @brief: Return a unit from the pool.
     *
     * @return: Shared pointer to a @ref MediaUnit, or nullptr if no media units are available.
     */
    std::shared_ptr<MediaUnit> get_media_unit();
    /**
     * @brief: Return a media unit from the pool, blocking until one is available.
     *
     * @return: Shared pointer to a @ref MediaUnit.
     */
    std::shared_ptr<MediaUnit> get_media_unit_blocking();
    /**
     * @brief: Return the number of available media units in the pool.
     *
     * @return: Number of available media units.
     */
    size_t get_available_media_units_count() const;
    /**
     * @brief: Return the total number of media units in the pool.
     *
     * @return: Total number of media units.
     */
    size_t get_total_media_units_count() const { return m_media_unit_count; }
    /**
     * @brief: Stop the media unit pool and release all waiting threads.
     */
    void stop();
    /**
     * @brief: Return the memory type used for allocation.
     *
     * @return: Memory type used for allocation.
     */
    MemoryLocation get_memory_location() const { return m_memory_location; }
private:
    /**
     * @brief: Returns a media unit back to the pool.
     *
     * This method is called when a @ref MediaUnit is no longer in use and needs to be
     * returned to the pool of available media units. It ensures thread safety and notifies
     * any waiting threads that a media unit is available.
     *
     * @param [in] index: Index of the media unit to be returned to the pool.
     */
    void return_media_unit_to_pool(size_t index);
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_MEDIA_UNIT_POOL_H_ */
