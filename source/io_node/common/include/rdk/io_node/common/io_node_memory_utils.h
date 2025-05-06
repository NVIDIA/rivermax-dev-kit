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

#ifndef RDK_IO_NODE_COMMON_IO_NODE_MEMORY_UTILS_H_
#define RDK_IO_NODE_COMMON_IO_NODE_MEMORY_UTILS_H_

#include <vector>
#include <utility>
#include <memory>

#include "rdk/services/memory_allocation/memory_allocation.h"

using namespace rivermax::dev_kit::services;

namespace rivermax
{
namespace dev_kit
{
namespace io_node
{
/**
 * @brief: IONodeMemoryUtils class.
 *
 * This class provides utility functions for managing memory allocation and alignment for I/O nodes.
 */
class IONodeMemoryUtils
{
public:
    /**
     * @brief: Constructs IONodeMemoryUtils object.
     *
     * @param [in] header_allocator: Reference to the memory allocator for headers.
     * @param [in] payload_allocator: Reference to the memory allocator for payloads.
     */
    IONodeMemoryUtils(MemoryAllocator& header_allocator, MemoryAllocator& payload_allocator);
    /**
     * @brief: Returns a pair of aligned buffer sizes.
     *
     * This method aligns the given buffer sizes to the memory page size.
     *
     * @param [in] buffer_sizes: Pair representing buffer sizes.
     *
     * @return: Pair representing aligned buffer sizes.
    */
    std::pair<size_t, size_t> align_buffer_sizes(const std::pair<size_t, size_t>& buffer_sizes) const;
    /**
     * @brief: Builds aligned buffer sizes.
     *
     * This method takes a vector of buffer sizes and aligns them to the memory page size.
     *
     * @param [in] buffer_sizes: Vector of pairs representing buffer sizes.
     *
     * @return: Vector of pairs representing aligned buffer sizes.
     */
    std::vector<std::pair<size_t, size_t>> build_aligned_buffer_sizes(const std::vector<std::pair<size_t, size_t>>& buffer_sizes) const;
    /**
     * @brief: Returns aligned header size.
     *
     * This method aligns the given header size to the memory page size.
     *
     * @param [in] size: Size of the header.
     *
     * @return: Aligned header size.
     */
    size_t get_aligned_header_size(size_t size) const;
    /**
     * @brief: Returns aligned payload size.
     *
     * This method aligns the given payload size to the memory page size.
     *
     * @param [in] size: Size of the payload.
     *
     * @return: Aligned payload size.
     */
    size_t get_aligned_payload_size(size_t size) const;
    /**
     * @brief: Returns total memory requirements.
     *
     * This method calculates the total memory required for headers and payloads.
     *
     * @param [in] buffer_sizes: Vector of pairs representing buffer sizes.
     *
     * @return: Pair representing total memory required for headers and payloads.
     */
    std::pair<size_t, size_t> get_total_memory_requirements(const std::vector<std::pair<size_t, size_t>>& buffer_sizes) const;
    /**
     * @brief: Returns header memory utilities.
     *
     * This method returns a shared pointer to the memory utilities for headers.
     *
     * @return: Shared pointer to MemoryUtils for headers.
     */
    std::shared_ptr<MemoryUtils> get_header_memory_utils() const;
    /**
     * @brief: Returns payload memory utilities.
     *
     * This method returns a shared pointer to the memory utilities for payloads.
     *
     * @return: Shared pointer to MemoryUtils for payloads.
     */
    std::shared_ptr<MemoryUtils> get_payload_memory_utils() const;
private:
    MemoryAllocator& m_header_allocator;
    MemoryAllocator& m_payload_allocator;
};

} // namespace io_node
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_IO_NODE_COMMON_IO_NODE_MEMORY_UTILS_H_ */
