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

#ifndef RDK_SERVICES_ERROR_HANDLING_EXCEPTION_H_
#define RDK_SERVICES_ERROR_HANDLING_EXCEPTION_H_

#include <stdexcept>
#include <string>

namespace rivermax
{
namespace dev_kit
{
namespace services
{

constexpr const char* NOT_IMPLEMENTED_ERROR = "Not implemented";

/**
 * @brief: Base exception class.
 *
 * The base exception class for the library.
 * All future exceptions should inherit from this exception.
 */
class RivermaxDevKitException : public std::exception
{
private:
    std::string m_error;
public:
    RivermaxDevKitException(const std::string& error) : m_error(error) {}
    const char* what() const noexcept override { return m_error.c_str(); }
};

/**
 * @brief: Not implemented SW component exception.
 *
 * The exception will be used for functions that are not yet implemented
 * and might be implemented in the future.
 */
class NotImplementedException : public RivermaxDevKitException
{
public:
    NotImplementedException() : RivermaxDevKitException(NOT_IMPLEMENTED_ERROR) {};
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_ERROR_HANDLING_EXCEPTION_H_ */
