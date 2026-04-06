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

#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <cstring>

#include "rdk/services/utils/environment.h"

using namespace rdk::services;

#ifdef __linux__

/**
 * @brief: Linux environment utilities implementation.
 */
class LinuxEnvironmentUtilsImp : public EnvironmentUtilsImp
{
public:
    ReturnStatus set_variable(const std::string& name, const std::string& value) override
    {
        int status = setenv(name.c_str(), value.c_str(), 1);
        if (status != 0) {
            std::cerr << "Failed to set environment variable '" << name << "' with error: "
                      << strerror(errno) << std::endl;
            return ReturnStatus::failure;
        }
        std::cout << "Set environment variable '" << name << "' with value: " << value << std::endl;
        return ReturnStatus::success;
    }

    ReturnStatus get_variable(const std::string& name, std::string& value) override
    {
        const char* env_value = std::getenv(name.c_str());
        if (env_value == nullptr) {
            std::cerr << "Failed to get environment variable '" << name << "'" << std::endl;
            return ReturnStatus::failure;
        }
        value = env_value;
        return ReturnStatus::success;
    }
};

#elif defined(_WIN32)

/**
 * @brief: Windows environment utilities implementation.
 */
class WindowsEnvironmentUtilsImp : public EnvironmentUtilsImp
{
public:
    ReturnStatus set_variable(const std::string& name, const std::string& value) override
    {
        errno_t status = _putenv_s(name.c_str(), value.c_str());
        if (status != 0) {
            std::cerr << "Failed to set environment variable '" << name << "' with error: "
                      << strerror(status) << std::endl;
            return ReturnStatus::failure;
        }
        std::cout << "Set environment variable '" << name << "' with value: " << value << std::endl;
        return ReturnStatus::success;
    }

    ReturnStatus get_variable(const std::string& name, std::string& value) override
    {
        size_t required_size = 0;
        getenv_s(&required_size, nullptr, 0, name.c_str());
        if (required_size == 0) {
            std::cerr << "Failed to get environment variable '" << name << "'" << std::endl;
            return ReturnStatus::failure;
        }
        value.resize(required_size);
        errno_t status = getenv_s(&required_size, &value[0], required_size, name.c_str());
        if (status != 0) {
            std::cerr << "Failed to get environment variable '" << name << "' with error: "
                      << strerror(status) << std::endl;
            return ReturnStatus::failure;
        }
        // Remove null terminator from string
        if (!value.empty() && value.back() == '\0') {
            value.pop_back();
        }
        return ReturnStatus::success;
    }
};

#endif // __linux__ / _WIN32

ReturnStatus EnvironmentUtils::set_variable(const std::string& name, const std::string& value)
{
    return get_imp().set_variable(name, value);
}

ReturnStatus EnvironmentUtils::get_variable(const std::string& name, std::string& value)
{
    return get_imp().get_variable(name, value);
}

EnvironmentUtilsImp& EnvironmentUtils::get_imp()
{
#ifdef __linux__
    static LinuxEnvironmentUtilsImp imp;
#elif defined(_WIN32)
    static WindowsEnvironmentUtilsImp imp;
#else
    static_assert(false, "Unsupported platform");
#endif
    return imp;
}
