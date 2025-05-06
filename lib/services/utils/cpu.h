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

#ifndef RMAX_APPS_LIB_LIB_SERVICES_UTILS_CPU_H_
#define RMAX_APPS_LIB_LIB_SERVICES_UTILS_CPU_H_

#include <vector>

#include "services/error_handling/return_status.h"


namespace ral
{
namespace lib
{
namespace services
{

/**
* @brief: Sets CPU affinity for the rivermax internal thread.
*
* @param [in] cpu_affinity: CPU number to assign for the thread.
*
* @return: Status of the operation.
*/
ReturnStatus set_rivermax_thread_cpu_affinity(int cpu);
/**
* @brief: Sets CPU affinity for the current thread.
*
* @param [in] cpu: A CPU core number to assign to the current thread.
*/
void set_current_thread_affinity(const int cpu);

} // namespace services
} // namespace lib
} // namespace ral

#endif /* RMAX_APPS_LIB_LIB_SERVICES_UTILS_CPU_H_ */
