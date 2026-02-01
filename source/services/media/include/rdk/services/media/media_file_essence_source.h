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

#ifndef RDK_SERVICES_MEDIA_MEDIA_FILE_ESSENCE_SOURCE_H_
#define RDK_SERVICES_MEDIA_MEDIA_FILE_ESSENCE_SOURCE_H_

#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>

#include "rdk/services/media/media_essence_source.h"
#include "rdk/services/media/media_file_reader.h"
#include "rdk/services/media/media_settings.h"
#include "rdk/services/memory_allocation/memory_allocation.h"
#include "rdk/services/utils/defs.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 * @brief: Reads media units from a binary file.
 *
 * It loads media units into a queue and, if the loop option is enabled,
 * re-inserts media units after serving them.
 */
class MediaFileEssenceSource : public IMediaEssenceSource
{
protected:
    std::string m_file_path;
    MemoryAllocator& m_mem_allocator;
    SMPTEStandard m_smpte_standard;
    size_t m_media_unit_size;
    size_t m_aligned_media_unit_size;
    bool m_loop_media_units;
    bool m_stop;
    bool m_media_units_loaded;
    std::queue<std::shared_ptr<MediaUnit>> m_media_unit_queue;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    MediaFileReader m_file_reader;
public:
    /**
     * @brief: Constructor.
     *
     * @param [in] file_path: Path to the media file.
     * @param [in] smpte_standard: SMPTE standard.
     * @param [in] media_unit_size: Size of each media unit.
     * @param [in] mem_allocator: Memory allocator to use.
     * @param [in] loop: Whether to loop media units.
     */
    MediaFileEssenceSource(const std::string& file_path, SMPTEStandard smpte_standard, size_t media_unit_size,
                           MemoryAllocator& mem_allocator, bool loop = false);
    /**
     * @brief: Destructor.
     */
    ~MediaFileEssenceSource();
    std::shared_ptr<MediaUnit> get_media_unit_blocking() override;
    std::shared_ptr<MediaUnit> get_media_unit_non_blocking() override;
    void stop() override;
    ReturnStatus get_data_size(size_t& data_size) const override;
    /**
     * @brief: Loads media units from the file.
     */
    ReturnStatus load_media_units();
private:
    /**
     * @brief: Handles looping of media units.
     *
     * @param [in] media_unit: Media unit to handle.
     */
    void handle_looping_media_unit(std::shared_ptr<MediaUnit>& media_unit);
    /**
     * @brief: Allocates memory for the media units.
     *
     * @param [in] file_size: Size of the file.
     * @param [out] file_memory_buffer: Pointer to the allocated memory.
     * @param [out] required_memory_size: Size of the allocated memory.
     *
     * @return: Status of the operation.
     */
    ReturnStatus allocate_media_units_memory(size_t file_size, byte_t*& file_memory_buffer,
                                             size_t& required_memory_size);
    /**
     * @brief: Reads media units from the file and puts them in the queue.
     *
     * @param [in] file_memory_buffer: Pointer to the memory buffer.
     *
     * @return: Status of the operation.
     */
    ReturnStatus read_media_units(byte_t* file_memory_buffer);
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_MEDIA_FILE_ESSENCE_SOURCE_H_ */
