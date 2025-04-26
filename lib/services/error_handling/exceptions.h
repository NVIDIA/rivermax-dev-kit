/*
 * Copyright © 2017-2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
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
    /**
     * @brief: Returns the exception error.
     *
     * This is a general exception for the library.
     *
     * @return: C string with the exception error.
     */
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
