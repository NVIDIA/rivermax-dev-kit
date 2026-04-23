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

#ifndef RDK_SERVICES_CPU_AFFINITY_RIVERMAX_AFFINITY_H_
#define RDK_SERVICES_CPU_AFFINITY_RIVERMAX_AFFINITY_H_

#include <cstdint>
#include <cstring>
#include <vector>

#include "rdk/services/cpu/affinity/rivermax_os_affinity.h"

namespace rdk
{
namespace services
{

/** Sentinel value indicating no CPU affinity is requested. */
constexpr int NO_CPU_AFFINITY = -1;

/**
 * @brief: Bitmask representing a set of CPU cores.
 *
 * Each bit corresponds to one CPU core. Bit @c i in word @c i/bits_per_word
 * represents core @c i. The layout is compatible with the @c uint64_t* buffers
 * accepted by @c rmx_set_cpu_affinity / @c rmx_mark_cpu_for_affinity (declared
 * in @c <rivermax_api.h>), so @c bits can be passed to those APIs directly.
 */
struct cpu_mask_t {
    static constexpr size_t max_cpus = 1024;
    static constexpr size_t bits_per_word = sizeof(uint64_t) * 8;
    uint64_t bits[max_cpus / bits_per_word] = {};

    constexpr cpu_mask_t() = default;
    /**
     * @brief: Constructs a mask with @p value in the lowest-order word.
     *
     * Sets @c bits[0] to @p value and zeroes all remaining words. This
     * allows single-expression constant masks such as
     * @code constexpr cpu_mask_t m { 1ULL << 5 }; @endcode
     *
     * @param [in] value: Bitmask for CPU cores 0–63.
     */
    constexpr explicit cpu_mask_t(uint64_t value) : bits{value} {}
    /**
     * @brief: Constructs a mask by copying @c max_cpus / @c bits_per_word
     *         words from @p src.
     *
     * @param [in] src: Pointer to a buffer holding at least
     *                  @c max_cpus / @c bits_per_word @c uint64_t words.
     *                  Must not be @c nullptr.
     */
    explicit cpu_mask_t(const uint64_t* src) {
        std::memcpy(bits, src, sizeof(bits));
    }
};

/**
 * Exclusive upper bound on CPU core numbers, equal to the affinity mask
 * capacity. Valid core IDs are in the range @c [0, MAX_CPU_RANGE).
 */
constexpr int MAX_CPU_RANGE = static_cast<int>(cpu_mask_t::max_cpus);

/**
 * @brief: Manages CPU affinity for threads.
 *
 * Wraps the OS-specific affinity API and provides methods to pin a thread
 * to one or more CPU cores, identified either by a single processor number
 * or by a @ref cpu_mask_t bitmask.
 */
class Affinity : public OsSpecificAffinity
{
public:
    using mask = cpu_mask_t;

protected:
    static const os_api default_api;

public:
    Affinity(const os_api &os_api = default_api);
    ~Affinity();
    /**
     * @brief: Sets CPU affinity for the given thread to a single core.
     *
     * @param [in] thread: Thread to pin.
     * @param [in] processor: Zero-based CPU core number.
     */
    void set(std::thread &thread, const size_t processor);
    /**
     * @brief: Sets CPU affinity for the given thread to a set of cores.
     *
     * @param [in] thread: Thread to pin.
     * @param [in] cpu_mask: Bitmask of CPU cores to assign.
     */
    void set(std::thread &thread, const mask &cpu_mask);
    /**
     * @brief: Sets CPU affinity for the calling thread to a single core.
     *
     * @param [in] processor: Zero-based CPU core number.
     */
    void set(const size_t processor);
    /**
     * @brief: Sets CPU affinity for the calling thread to a set of cores.
     *
     * @param [in] cpu_mask: Bitmask of CPU cores to assign.
     */
    void set(const mask &cpu_mask);
private:
    void fill_with(const mask &cpu_mask, editor &editor);
};

/**
 * @brief: Sets CPU affinity for the calling thread to a single core.
 *
 * @param [in] processor: Zero-based CPU core number.
 *
 * @return: true on success, false on failure.
 */
bool set_affinity(const size_t processor) noexcept;

/**
 * @brief: Sets CPU affinity for the calling thread to a set of cores.
 *
 * @param [in] cpu_mask: Bitmask of CPU cores to assign.
 *
 * @return: true on success, false on failure.
 */
bool set_affinity(const Affinity::mask &cpu_mask) noexcept;

/**
* @brief: Validates that the requested CPU cores are within the process affinity mask.
*
* @param [in] internal_thread_affinity: CPU core for the internal Rivermax thread;
*                                       @ref NO_CPU_AFFINITY means no affinity.
* @param [in] cpus: CPU core numbers for application threads;
*                   @ref NO_CPU_AFFINITY entries are ignored.
*
* @return: true if all requested cores are valid, false otherwise.
*/
bool validate_thread_affinity_cpus(int internal_thread_affinity, const std::vector<int> &cpus);

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_CPU_AFFINITY_RIVERMAX_AFFINITY_H_ */
