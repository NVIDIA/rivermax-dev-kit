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

#ifndef RDK_SERVICES_UTILS_ENUM_H_
#define RDK_SERVICES_UTILS_ENUM_H_

#include <array>
#include <string>

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Hashing Enum class.
 *
 * This class can be used as a functor for class types that requires hashing for
 * STL operations.
 */
struct EnumClassHash
{
    /**
     * @brief: Hashing operator.
     *
     * @tparam EnumType: The Enum class type.
     *
     * @param [in] value: The Enum class value.
     *
     * @return: The hash value of the Enum class value.
     */
    template <typename EnumType>
    std::size_t operator()(EnumType value) const
    {
        return static_cast<std::size_t>(value);
    }
};
/**
 * @brief: Enum class mapper.
 *
 * This class is responsible for mapping Enum classes to their string names.
 *
 * @tparam EnumType: The Enum class type.
 */
template <typename EnumType>
class EnumMapper
{
public:
    /* The number of Enum values. */
    static constexpr size_t size = static_cast<size_t>(EnumType::Unknown);
    /* The array of Enum names. */
    static const std::array<const char*, size> names;
    /**
     * @brief: Returns the name of the given Enum value.
     *
     * @param [in] value: The Enum value.
     *
     * @return: The name of the Enum value.
     */
    static const char* name(EnumType value)
    {
        auto index = static_cast<size_t>(value);
        return (index < size) ? names[index] : "Unknown";
    }
};
/**
 * @brief: Enum name array.
 *
 * This template alias is used to define an array of Enum names.
 *
 * @tparam EnumType: The Enum class type.
 */
template <typename EnumType>
using EnumNameArray = const std::array<const char*, EnumMapper<EnumType>::size>;
/**
 * @brief: Converts the given Enum value to a string.
 *
 * @tparam EnumType: The Enum class type.
 *
 * @param [in] value: The Enum value.
 *
 * @return: The string representation of the Enum value.
 */
template <typename EnumType>
std::string enum_to_string(EnumType value)
{
    return EnumMapper<EnumType>::name(value);
}

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_UTILS_ENUM_H_ */
