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

#ifndef RDK_SERVICES_SDP_SDP_INTERFACE_H_
#define RDK_SERVICES_SDP_SDP_INTERFACE_H_

#include <memory>
#include <stdexcept>
#include <iostream>

#include "sdptransform.hpp"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Interface for constructing SDP following RFC4566.
 */
class ISDP
{
public:
    virtual ~ISDP() = default;
    /**
     * @brief: Converts the object to a JSON object implicitly.
     *
     * @return: The JSON object.
     */
    virtual operator json() const = 0;
    /**
     * @brief: Interface for constructing SDP builders.
     *
     * @tparam ConcreteSessionDescription: The concrete session description object type.
     * @tparam ConcreteBuilder: The concrete builder object type.
     */
    template <typename ConcreteSessionDescription, typename ConcreteBuilder>
    class IBuilder
    {
    public:
        /**
         * @brief: Builds and returns a fully constructed ISDP object.
         *
         * @return: A constructed ISDP object.
         */
        std::unique_ptr<ConcreteSessionDescription> build() { return std::move(m_instance); }
        /**
         * @brief: Throws an error if mandatory fields are missing.
         *
         * @param [in] condition: The condition to check.
         */
        static void throw_if(bool condition)
        {
            if (condition) {
                auto error_message = "Builder mandatory fields are missing.";
                std::cerr << error_message << std::endl;
                throw std::runtime_error(error_message);
            }
        }

    protected:
        IBuilder() : m_instance(new ConcreteSessionDescription()) {}
        virtual ~IBuilder() = default;
        /**
         * @brief: Sets the member variable with the given value.
         *
         * @tparam ParameterType: The parameter type.
         * @tparam ValueType: The value type.
         *
         * @param [in] member: The member variable.
         * @param [in] value: The value to set.
         *
         * @return: Reference to the builder object.
         */
        template <typename ParameterType, typename ValueType>
        ConcreteBuilder& set(ParameterType& member, ValueType&& value)
        {
            member = std::forward<ValueType>(value);
            return static_cast<ConcreteBuilder&>(*this);
        }

        std::unique_ptr<ConcreteSessionDescription> m_instance;
    };

protected:
    ISDP() = default;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_SDP_SDP_INTERFACE_H_ */
