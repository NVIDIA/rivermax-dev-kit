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

#ifndef RMAX_APPS_LIB_LIB_SERVICES_ERROR_HANDLING_RETURN_STATUS_H_
#define RMAX_APPS_LIB_LIB_SERVICES_ERROR_HANDLING_RETURN_STATUS_H_

namespace ral
{
namespace lib
{
namespace services
{

/**
 * @brief: Return status ENUM class.
 *
 * The return status will be used to pass function call exit status
 * between the different components of the library.
 */
enum class ReturnStatus
{
    success,
    failure,
    success_cli_help,
    signal_received,
    obj_init_success,
    obj_init_failure,
    hw_send_queue_full,
    memory_allocation_failure,
    rmax_version_incompatible,
    rmax_version_unaligned,
    no_free_chunks,
    no_completion,
};

} // namespace services
} // namespace lib
} // namespace ral

#endif /* RMAX_APPS_LIB_LIB_SERVICES_ERROR_HANDLING_RETURN_STATUS_H_ */
