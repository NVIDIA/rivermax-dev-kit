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

#ifndef RDK_SERVICES_UTILS_STATIC_RCU_H_
#define RDK_SERVICES_UTILS_STATIC_RCU_H_

#include <cstddef>
#include <atomic>
#include <array>

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 * @brief: RCU template class using static memory allocation.
 *
 * This class implements a Read-Copy-Update (RCU) synchronization mechanism
 * with a static memory allocation for the shared data versions.
 */
template <typename ValueType, size_t num_versions>
class StaticRCU {
public:
    /**
     * @brief: StaticRCU constructor.
     */
    StaticRCU() : m_current_index(0)
    {
        for (size_t i = 0; i < num_versions; ++i) {
            m_versions[i].store(&m_buffer[i], std::memory_order_relaxed);
        }
    }
    /**
     * @brief: Reads the RCU most recent version of the value.
     *
     * @return: A reference to the most recent version of the shared data.
     */
    ValueType& read() const
    {
        return *m_versions[m_current_index.load(std::memory_order_acquire)].load(std::memory_order_acquire);
    }
    /**
     * @brief: Updates the RCU value with a new version.
     *
     * @param [in] new_data: A reference to the new version of the shared data to be stored.
     */
    void update(const ValueType& new_data)
    {
        size_t new_index = (m_current_index.load(std::memory_order_relaxed) + 1) % num_versions;
        m_buffer[new_index] = new_data;
        m_versions[new_index].store(&m_buffer[new_index], std::memory_order_release);
        m_current_index.store(new_index, std::memory_order_release);
    }
private:
    std::array<ValueType, num_versions> m_buffer;
    std::array<std::atomic<ValueType*>, num_versions> m_versions;
    std::atomic<size_t> m_current_index;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif // RDK_SERVICES_UTILS_STATIC_RCU_H_
