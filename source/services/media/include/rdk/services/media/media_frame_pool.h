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

#ifndef RDK_SERVICES_MEDIA_MEDIA_FRAME_POOL_H_
#define RDK_SERVICES_MEDIA_MEDIA_FRAME_POOL_H_

#include <memory>
#include <mutex>
#include <queue>
#include <vector>
#include <condition_variable>
#include <atomic>

#include "rdk/services/utils/defs.h"
#include "rdk/services/media/media_frame_provider.h"
#include "rdk/services/memory_allocation/memory_allocation.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Manages a pool of pre-allocated @ref MediaFrame objects.
 *
 * This class allocates a contiguous block of memory and divides it into
 * fixed-size frames. Frames can be borrowed from the pool and are automatically
 * returned when no longer in use. The pool ensures efficient memory usage and
 * provides thread-safe access to the frames.
 *
 * The @ref MediaFramePool class supports both non-blocking and blocking methods for
 * obtaining frames. The non-blocking method returns a frame if available, or
 * nullptr if no frames are available. The blocking method waits until a frame
 * becomes available.
 *
 * The class also provides methods to query the number of available frames and
 * the total number of frames in the pool.
 */
class MediaFramePool {
private:
    /* The total memory block that contains all frames */
    byte_t* m_memory_block;
    /* Vector of all frames for tracking */
    std::vector<MediaFrame> m_all_frames;
    /* Queue of available frames */
    std::queue<size_t> m_available_indices;
    /* Mutex for thread safety */
    mutable std::mutex m_mutex;
    /* Condition variable for blocking operations */
    std::condition_variable m_cv;
    /* Size of each frame */
    size_t m_frame_size;
    /* Total number of frames */
    size_t m_frame_count;
    /* Total allocated memory size */
    size_t m_total_memory_size;
    /* Flag that signals that object destruction has been started */
    std::atomic<bool> m_in_destruction{false};
    /* Flag to indicate if the pool is stopping */
    std::atomic<bool> m_stop{false};
public:
    /**
     * @brief: Constructor.
     *
     * @param [in] frame_count: Number of frames to allocate in the pool.
     * @param [in] frame_size: Size of each frame in bytes.
     * @param [in] mem_allocator: Memory allocator to use.
     */
    MediaFramePool(size_t frame_count, size_t frame_size, MemoryAllocator& mem_allocator);
   /**
     * @brief: Constructor for external memory.
     *
     * @param [in] frame_count: Number of frames in the pool.
     * @param [in] frame_size: Size of each frame in bytes.
     * @param [in] memory_block: Pointer to the external memory block.
     * @param [in] memory_size: Size of the external memory block in bytes.
     *
     * @throws std::invalid_argument if the provided memory is insufficient.
     */
    MediaFramePool(size_t frame_count, size_t frame_size, byte_t* memory_block, size_t memory_size);
    /**
     * @brief: Destructor.
     */
    virtual ~MediaFramePool();
    /**
     * @brief: Return a frame from the pool.
     *
     * @return: Shared pointer to a @ref MediaFrame, or nullptr if no frames are available.
     */
    std::shared_ptr<MediaFrame> get_frame();
    /**
     * @brief: Return a frame from the pool, blocking until one is available.
     *
     * @return: Shared pointer to a @ref MediaFrame.
     */
    std::shared_ptr<MediaFrame> get_frame_blocking();
    /**
     * @brief: Return the number of available frames in the pool.
     *
     * @return: Number of available frames.
     */
    size_t get_available_frames_count() const;
    /**
     * @brief: Return the total number of frames in the pool.
     *
     * @return: Total number of frames.
     */
    size_t get_total_frames_count() const { return m_frame_count; }
    /**
     * @brief: Stop the frame pool and release all waiting threads.
     */
    void stop();
private:
   /**
     * @brief: Returns a frame back to the pool.
     *
     * This method is called when a @ref MediaFrame is no longer in use and needs to be
     * returned to the pool of available frames. It ensures thread safety and notifies
     * any waiting threads that a frame is available.
     *
     * @param [in] index: Index of the frame to be returned to the pool.
     */
    void return_frame_to_pool(size_t index);
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_MEDIA_FRAME_POOL_H_ */
