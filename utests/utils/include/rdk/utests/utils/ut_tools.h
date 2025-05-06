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

#ifndef RDK_UTESTS_UTILS_UT_TOOLS_H
#define RDK_UTESTS_UTILS_UT_TOOLS_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace rivermax
{
namespace dev_kit
{
namespace utests
{
template <typename T, size_t N>
constexpr size_t len(const T (&array)[N]) { return N; }

template <typename T, size_t N = 1>
void fill_with_random(T *data) {
    uint8_t *first = reinterpret_cast<uint8_t*>(data);
    uint8_t *end = first + sizeof(T) * N;
    for(auto next=first; next < end; ++next) {
        *next = rand();
    }
}

template <class T, size_t N=sizeof(T)>
struct comparable {
    const uint8_t &_raw_bytes;
    comparable(const T &object): _raw_bytes(reinterpret_cast<const uint8_t&>(object)) {}
    bool operator==(const T &other) const { return _raw_bytes == comparable(other)._raw_bytes; }
};

template <class T>
const comparable<T> to_comparable(const T&object) { return comparable<T>(object); }

} // namespace utests
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_UTESTS_UTILS_UT_TOOLS_H */
