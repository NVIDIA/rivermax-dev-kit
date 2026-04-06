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

#ifndef RDK_SERVICES_UTILS_DEFS_H_
#define RDK_SERVICES_UTILS_DEFS_H_

#include <cstdint>
#include <functional>

namespace rdk
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
 * @brief: Macro to check a condition and return a status with an error message.
 *
 * Evaluates the condition, and if true, displays the error message and returns the specified status.
 * Useful for early exits in functions with error handling. Meant to be used mainly
 * in the examples.
 *
 * @param [in] condition: Condition to evaluate.
 * @param [in] message: Error message to display if the condition is true.
 * @param [in] status: Status to return if the condition is true.
 */
#define RETURN_STATUS_IF(condition, message, status) \
    do {                                             \
        if (condition) {                             \
            std::cerr << message << std::endl;       \
            return status;                           \
        }                                            \
    } while (0)
/**
 * @brief: Macro to check if a status is not successful and return with failure status.
 *
 * Evaluates if the status is not equal to @ref ReturnStatus::success, and if true,
 * displays the error message and returns @ref ReturnStatus::failure. Meant to be used mainly
 * in the examples.
 *
 * @param [in] condition: Status variable to check.
 * @param [in] message: Error message to display if the status is not successful.
 */
#define RETURN_FAILURE_ON_ERROR(condition, message) RETURN_STATUS_IF( \
    condition != ReturnStatus::success, message, ReturnStatus::failure)
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
    GPU,
    GPUHostPinned,
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
/**
 * @brief: Byte memory literal operator.
 *
 * In this operator, the value is treated as bytes, but no conversion is done.
 * The value is returned as-is. Useful for defining byte values in a more readable way.
 *
 * @param [in] value: Value to convert to bytes.
 *
 * @return: Value in bytes.
 */
constexpr size_t operator"" _B(unsigned long long value) { return value; }
/**
 * @brief: Kilobyte memory literal operator.
 *
 * @param [in] value: Value to convert to kilobytes.
 *
 * @return: Value in kilobytes.
 */
constexpr size_t operator"" _KB(unsigned long long value) { return value << 10; }
/**
 * @brief: Megabyte memory literal operator.
 *
 * @param [in] value: Value to convert to megabytes.
 *
 * @return: Value in megabytes.
 */
constexpr size_t operator"" _MB(unsigned long long value) { return value << 20; }
/**
 * @brief: Gigabyte memory literal operator.
 *
 * @param [in] value: Value to convert to gigabytes.
 *
 * @return: Value in gigabytes.
 */
constexpr size_t operator"" _GB(unsigned long long value) { return value << 30; }

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_UTILS_DEFS_H_ */
