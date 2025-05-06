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

#ifndef RDK_SERVICES_UTILS_DEFS_H_
#define RDK_SERVICES_UTILS_DEFS_H_

#include <cstdint>
#include <functional>

namespace rivermax
{
namespace dev_kit
{
namespace services
{

#ifdef __GNUC__
#define likely(condition) __builtin_expect(static_cast<bool>(condition), 1)
#define unlikely(condition) __builtin_expect(static_cast<bool>(condition), 0)
#else
#define likely(condition) (condition)
#define unlikely(condition) (condition)
#endif
#define NOT_IN_USE(a) ((void)(a))
#define align_down_pow2(_n, _alignment) \
    ( (_n) & ~((_alignment) - 1) )
#define align_up_pow2(_n, _alignment) \
    align_down_pow2((_n) + (_alignment) - 1, _alignment)

typedef uint8_t byte_t;

/**
 * @brief: Packet pacing rate.
 */
struct PacketPacingRate
{
    uint64_t bps;
    uint32_t max_burst_in_packets;
};
/**
 * @brief: Allocator types for UI.
 */
enum class AllocatorTypeUI {
    Auto,
    Malloc,
    HugePageDefault,
    HugePage2MB,
    HugePage512MB,
    HugePage1GB,
    Gpu
};

/**
 * @brief: Time handler callback definition.
 *
 * This defines the callback type for time handling callback functions.
 */
typedef std::function<uint64_t(void*)> time_handler_ns_cb_t;
template <typename T, size_t S>
void set_bitmap_bit(T (&bitmap)[S], size_t bit)
{
    constexpr auto bit_size = sizeof(T) * 8;
    auto index = bit / bit_size;
    auto offset = bit % bit_size;
    bitmap[index] |= ((T)1) << offset;
}

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_UTILS_DEFS_H_ */
