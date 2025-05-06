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

#ifndef RDK_UTESTS_UTILS_BITMAP_H
#define RDK_UTESTS_UTILS_BITMAP_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <memory>
#include <stdexcept>

namespace rivermax
{
namespace dev_kit
{
namespace utests
{
class Bitmap {
    uint64_t *_bitmap;
    size_t _bit_count;
    std::unique_ptr<uint64_t[]> _memory;
public:
    Bitmap() = delete;
    Bitmap(size_t bit_count): _bit_count {bit_count} {
        _memory = std::make_unique<uint64_t[]>(size());
        _bitmap = _memory.get();
        clear();
    }
    Bitmap(uint64_t *bitmap, size_t bit_count):
        _bitmap {bitmap}, _bit_count {bit_count} {}

    bool operator[](size_t bit) const {
        if (bit >= _bit_count) { return false; }
        auto base = bit / 64;
        auto offset = bit % 64;
        return (_bitmap[base] & (1 << offset)) != 0;
    }

    void set(size_t bit) {
        if (bit >= _bit_count) { throw std::out_of_range("out of range bit index"); }
        auto base = bit / 64;
        auto offset = bit % 64;
        _bitmap[base] |= (1ULL << offset);

    }

    void clear(size_t bit) {
        if (bit >= _bit_count) { throw std::out_of_range("out of range bit index"); }
        auto base = bit / 64;
        auto offset = bit % 64;
        _bitmap[base] &= ~(1ULL << offset);
    }

    size_t size() const { return (_bit_count + 63) / 64 * sizeof(uint64_t); }

    void clear() {
        std::memset(_bitmap, 0, size());
    }

    bool operator==(const uint64_t *other) const {
        return std::memcmp(_bitmap, other, size()) == 0;
    }

    bool operator==(const Bitmap &other) const {
        return (_bit_count == other._bit_count) && (*this) == other._bitmap;
    }

    friend std::ostream& operator<<(std::ostream& os, const Bitmap& bitmap) {
        auto bytes = reinterpret_cast<const uint8_t*>(bitmap._bitmap);
        os << std::endl;
        for (auto i = 0; i < bitmap.size(); i++) {
            os << std::hex << std::setw(16) << std::setfill((bitmap._bitmap[i])? '0': '-') << bitmap._bitmap[i] << " ";
        }
        os << std::endl;
        return os;
    }
};

} // namespace utests
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_UTESTS_UTILS_BITMAP_H */
