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

#ifndef RDK_SERVICES_UTILS_CLOCK_H_
#define RDK_SERVICES_UTILS_CLOCK_H_

#include <string>

#include <rivermax_api.h>

#include "rdk/services/error_handling/return_status.h"

namespace rivermax
{
namespace dev_kit
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
/**
 * @brief: Returns Rivermax PTP time.
 *
 * Prior to calling this function, the PTP clock has to be configured with @ref set_rivermax_ptp_clock.
 *
 * @param [out] ptp_time_ns: PTP time in nanoseconds.
 *
 * @return: Status of the operation.
 */
ReturnStatus get_rivermax_ptp_time_ns(uint64_t& ptp_time_ns);
/**
 * @brief: Converts PTP timestamp to a string.
 *
 * @param [in] timestamp: PTP timestamp.
 *
 * @return: Time string in "YYYY-mm-dd H:MM:SS.nnnnnnnnn" format.
 */
std::string ptp_to_string(uint64_t timestamp);
} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_UTILS_CLOCK_H_ */
