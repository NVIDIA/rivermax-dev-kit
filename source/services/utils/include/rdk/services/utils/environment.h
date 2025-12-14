/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef RDK_SERVICES_UTILS_ENVIRONMENT_H_
#define RDK_SERVICES_UTILS_ENVIRONMENT_H_

#include <string>

#include "rdk/services/error_handling/return_status.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Environment utilities implementation interface.
 *
 * The base class for cross-platform environment variable operations.
 * Platform-specific implementations should derive from this class.
 */
class EnvironmentUtilsImp
{
public:
    EnvironmentUtilsImp() = default;
    virtual ~EnvironmentUtilsImp() = default;
    /**
     * @brief: Sets an environment variable.
     *
     * @param [in] name: The name of the environment variable.
     * @param [in] value: The value to set.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus set_variable(const std::string& name, const std::string& value) = 0;
    /**
     * @brief: Returns an environment variable value.
     *
     * @param [in] name: The name of the environment variable.
     * @param [out] value: The retrieved value.
     *
     * @return: Status of the operation. Returns failure if variable is not set.
     */
    virtual ReturnStatus get_variable(const std::string& name, std::string& value) = 0;
};

/**
 * @brief: Environment utilities.
 *
 * Provides cross-platform environment variable operations.
 * Uses the implementation idiom to delegate to OS-specific implementations.
 * All methods are static for convenience since environment variables are global.
 */
class EnvironmentUtils
{
public:
    EnvironmentUtils() = delete;
    /**
     * @brief: Sets an environment variable.
     *
     * @param [in] name: The name of the environment variable.
     * @param [in] value: The value to set.
     *
     * @return: Status of the operation.
     */
    static ReturnStatus set_variable(const std::string& name, const std::string& value);
    /**
     * @brief: Returns an environment variable value.
     *
     * @param [in] name: The name of the environment variable.
     * @param [out] value: The retrieved value.
     *
     * @return: Status of the operation. Returns failure if variable is not set.
     */
    static ReturnStatus get_variable(const std::string& name, std::string& value);

private:
    /**
     * @brief: Returns OS-specific EnvironmentUtilsImp singleton.
     *
     * @return: Reference to the OS-specific implementation.
     */
    static EnvironmentUtilsImp& get_imp();
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_UTILS_ENVIRONMENT_H_ */
