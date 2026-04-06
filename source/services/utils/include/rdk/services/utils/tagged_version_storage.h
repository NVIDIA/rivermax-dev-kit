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

#ifndef RDK_SERVICES_UTILS_TAGGED_VERSION_STORAGE_H_
#define RDK_SERVICES_UTILS_TAGGED_VERSION_STORAGE_H_

#include <array>
#include <atomic>
#include <cstddef>

#include "rdk/services/error_handling/return_status.h"

namespace rdk
{
namespace services
{

/**
 * @brief: Storage for tagged versions of data with fixed capacity.
 *
 * This class provides thread-safe storage for multiple versions of data,
 * where each version is associated with a tag. The storage maintains
 * a fixed number of most recent versions.
 *
 * @tparam ValueType: The type of data to store.
 * @tparam TagType: The type to use as a tag (must be equality comparable).
 * @tparam num_versions: The maximum number of versions to store.
 */
template <typename ValueType, typename TagType, std::size_t num_versions>
class TaggedVersionStorage {
public:
    /**
     * @brief: TaggedVersionStorage constructor.
     */
    TaggedVersionStorage() : m_current_index(0) {
        for (auto& used : m_used) {
            used.store(false, std::memory_order_relaxed);
        }
    }
    /**
     * @brief: Adds a new version with a tag.
     *
     * @param [in] tag: The tag to associate with the version.
     * @param [in] value: The value to store.
     */
    void add_version(const TagType& tag, const ValueType& value) {
        std::size_t index = m_current_index.load(std::memory_order_relaxed);
        std::size_t next_index = (index + 1) % num_versions;
        m_versions[next_index] = value;
        m_tags[next_index] = tag;
        m_used[next_index].store(true, std::memory_order_release);
        m_current_index.store(next_index, std::memory_order_release);
        return;
    }
    /**
     * @brief: Retrieves the most recent version.
     *
     * @param [out] value: The value to store the result in.
     *
     * @return: Status of the operation.
     */
    ReturnStatus retrieve_latest(ValueType& value) const {
        std::size_t index = m_current_index.load(std::memory_order_acquire);
        if (!m_used[index].load(std::memory_order_acquire)) {
            return ReturnStatus::failure;
        }
        value = m_versions[index];
        return ReturnStatus::success;
    }
    /**
     * @brief: Retrieves a version by its tag.
     *
     * @param [in] tag: The tag to search for.
     * @param [out] value: The value to store the result in.
     *
     * @return: Status of the operation.
     */
    ReturnStatus retrieve_by_tag(const TagType& tag, ValueType& value) const {
        std::size_t index = m_current_index.load(std::memory_order_acquire);
        constexpr std::size_t max_search_steps = num_versions - 1;
        for (std::size_t i = 0; i < max_search_steps; ++i) {
            std::size_t current = (index + num_versions - i) % num_versions;
            if (!m_used[current].load(std::memory_order_acquire)) {
                break;
            }
            if (m_tags[current] == tag) {
                value = m_versions[current];
                return ReturnStatus::success;
            }
        }
        return ReturnStatus::failure;
    }
    /**
     * @brief: Retrieves the tag of the most recent version.
     *
     * @param [out] tag: The tag to store the result in.
     *
     * @return: Status of the operation.
     */
    ReturnStatus retrieve_latest_tag(TagType& tag) const {
        std::size_t index = m_current_index.load(std::memory_order_acquire);
        if (!m_used[index].load(std::memory_order_acquire)) {
            return ReturnStatus::failure;
        }
        tag = m_tags[index];
        return ReturnStatus::success;
    }
    /**
     * @brief: Checks if a version with the given tag exists.
     *
     * @param [in] tag: The tag to check.
     *
     * @return: Status of the operation.
     */
    ReturnStatus has_tag(const TagType& tag) const {
        ValueType dummy;
        return retrieve_by_tag(tag, dummy);
    }
    /**
     * @brief: Clears all versions.
     */
    void clear() {
        for (auto& used : m_used) {
            used.store(false, std::memory_order_release);
        }
        m_current_index.store(0, std::memory_order_release);
    }

private:
    std::array<ValueType, num_versions> m_versions;
    std::array<TagType, num_versions> m_tags;
    std::array<std::atomic<bool>, num_versions> m_used;
    std::atomic<std::size_t> m_current_index;
};

} // namespace services
} // namespace rdk

#endif // RDK_SERVICES_UTILS_TAGGED_VERSION_STORAGE_H_
