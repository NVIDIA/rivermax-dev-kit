/*
 * Copyright © 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
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
* @param [in] cpu_affinity: Vector of CPU numbers to assign for the thread.
*
* @return: Status of the operation.
*/
ReturnStatus set_rivermax_thread_cpu_affinity(const std::vector<int>& cpu_affinity);
/**
 * @brief: Sets CPU affinity for the current thread.
 *
 * @param [in] cpu_affinity: Vector of CPU numbers to assign for the thread.
 *
 * @return: Status of the operation.
 */
ReturnStatus set_app_thread_cpu_affinity(const std::vector<int>& cpu_affinity);

} // namespace services
} // namespace lib
} // namespace ral

#endif /* RMAX_APPS_LIB_LIB_SERVICES_UTILS_CPU_H_ */
