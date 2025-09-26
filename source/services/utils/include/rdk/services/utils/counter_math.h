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

#ifndef RDK_SERVICES_UTILS_COUNTER_MATH_H_
#define RDK_SERVICES_UTILS_COUNTER_MATH_H_

#include <cstdint>

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: A helper function to compare uint32_t counter values (e.g. RTP timestamps) with care to wrap-around.
 *
 * A is considered before B if there is a X: 0 <= X <= 0x7fffffff such that B = (A + X) mod 2^32.
 * Or in other words when distance from A to B by counting modulo 2^32 is lesss than from B to A.
 *
 * @param [in] a: The first argument of "is before" comparison.
 * @param [in] b: The second argument of "is before" comparison.
 *
 * @return: True if a is before b.
 */
static inline bool is_counter_before(uint32_t a, uint32_t b) {
    return (((a & 0x80000000) == (b & 0x80000000)) && (a < b)) ||
           (((a & 0x80000000) != (b & 0x80000000)) && ((a & 0x7fffffff) > (b & 0x7fffffff)));
}

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_UTILS_COUNTER_MATH_H_ */
