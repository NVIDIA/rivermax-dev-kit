/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef RMAX_APPS_LIB_LIB_SERVICES_ERROR_HANDLING_EXCEPTION_H_
#define RMAX_APPS_LIB_LIB_SERVICES_ERROR_HANDLING_EXCEPTION_H_

#include <stdexcept>
#include <string>

namespace ral
{
namespace lib
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
class RmaxAppsLibException : public std::exception
{
private:
    std::string m_error;
public:
    RmaxAppsLibException(const std::string& error) : m_error(error) {}
    const char* what() const noexcept override { return m_error.c_str(); }
};

/**
 * @brief: Not implemented SW component exception.
 *
 * The exception will be used for functions that are not yet implemented
 * and might be implemented in the future.
 */
class NotImplementedException : public RmaxAppsLibException
{
public:
    NotImplementedException() : RmaxAppsLibException(NOT_IMPLEMENTED_ERROR) {};
};

} // namespace services
} // namespace lib
} // namespace ral

#endif /* RMAX_APPS_LIB_LIB_SERVICES_ERROR_HANDLING_EXCEPTION_H_ */
