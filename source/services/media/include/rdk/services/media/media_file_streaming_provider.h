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

#ifndef RDK_SERVICES_MEDIA_MEDIA_FILE_STREAMING_PROVIDER_H_
#define RDK_SERVICES_MEDIA_MEDIA_FILE_STREAMING_PROVIDER_H_

#include <atomic>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <string>
#include <thread>

#include "rdk/services/media/media_file_reader.h"
#include "rdk/services/memory_allocation/memory_allocation.h"
#include "rdk/services/utils/defs.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
class MediaUnitPool;
class BufferedEssenceSource;
/**
 * @brief: Reads media units from a file and streams them to a @ref BufferedEssenceSource.
 *
 * This class reads media units from a file, uses a @ref MediaUnitPool for memory management,
 * and pushes the media units to a @ref BufferedEssenceSource for consumption by other components.
 * It supports looping through the file when reaching the end and provides thread-safe
 * operations for starting and stopping the streaming process.
 */
class MediaFileStreamingProvider {
private:
    std::string m_file_path;
    SMPTEStandard m_smpte_standard;
    size_t m_media_unit_size;
    std::unique_ptr<MediaUnitPool> m_media_unit_pool;
    std::shared_ptr<BufferedEssenceSource> m_essence_source;
    std::shared_ptr<MemoryUtils> m_memory_utils;
    std::shared_ptr<MemoryAllocator> m_memory_allocator;
    bool m_loop_media_units;
    std::atomic<bool> m_stop{false};
    std::condition_variable m_cv;
    std::mutex m_mutex;
    std::atomic<bool> m_initialized{false};
    size_t m_sleep_duration_microseconds;
    MediaFileReader m_file_reader;
    static constexpr auto SLEEP_DURATION_MICROSECONDS = 10000;
    static constexpr auto MEMORY_POOL_MEDIA_UNIT_COUNT = 15;
public:
    /**
     * @brief: Constructor.
     *
     * @param [in] file_path: Path to the media file.
     * @param [in] smpte_standard: SMPTE standard.
     * @param [in] media_unit_size: Size of each media unit in bytes.
     * @param [in] essence_source: Shared pointer to a @ref BufferedEssenceSource.
     * @param [in] memory_allocator: Shared pointer to a @ref MemoryAllocator.
     * @param [in] loop: Whether to loop through the file when reaching the end.
     * @param [in] sleep_duration_microseconds: Sleep duration in microseconds between reading media units.
     */
    MediaFileStreamingProvider(const std::string& file_path, SMPTEStandard smpte_standard,
                               size_t media_unit_size,
                               std::shared_ptr<BufferedEssenceSource> essence_source,
                               std::shared_ptr<MemoryAllocator> memory_allocator, bool loop = false,
                               size_t sleep_duration_microseconds = SLEEP_DURATION_MICROSECONDS);
    /**
     * @brief: Destructor.
     */
    virtual ~MediaFileStreamingProvider();
    /**
     * @brief: Initialize the streaming process.
     *
     * @return: Status of the operation.
     */
    ReturnStatus initialize();
    /**
     * @brief: Stop the streaming process.
     */
    void stop();
    /**
     * @brief: Call operator for running in a separate thread.
     *
     * This function runs in a separate thread, reads media units from the file,
     * and pushes them to the @ref BufferedEssenceSource. It handles looping
     * through the file if specified and ensures thread-safe operations.
     */
    void operator()();
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif // RDK_SERVICES_MEDIA_MEDIA_FILE_STREAMING_PROVIDER_H_
