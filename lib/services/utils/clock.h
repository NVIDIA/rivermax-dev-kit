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

#ifndef RMAX_APPS_LIB_LIB_SERVICES_UTILS_CLOCK_H_
#define RMAX_APPS_LIB_LIB_SERVICES_UTILS_CLOCK_H_

#include <rivermax_api.h>

#include "services/error_handling/return_status.h"


namespace ral
{
namespace lib
{
namespace services
{

/**
 * @brief: Sets Rivermax user clock.
 *
 * @param [in] handler: Rivermax clock handler, see @ref rmx_set_user_clock_handler in rivermax_api.h.
 * @param [in] ctx: Clock handler context, see @ref rmx_set_user_clock_context in rivermax_api.h.
 *
 * @return: Status of the operation.
 */
ReturnStatus set_rivermax_user_clock(rmx_user_clock_handler handler, void* ctx = nullptr);
/**
 * @brief: Sets Rivermax user clock.
 *
 * @param [in] device_iface: A device interface obtained with @ref rmx_retrieve_device_iface.
 *
 * @return: Status of the operation.
 */
ReturnStatus set_rivermax_ptp_clock(const rmx_device_iface* device_iface);

} // namespace services
} // namespace lib
} // namespace ral

#endif /* RMAX_APPS_LIB_LIB_SERVICES_UTILS_CLOCK_H_ */
