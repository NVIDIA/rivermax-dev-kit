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

#ifndef RDK_SERVICES_CPU_AFFINITY_AFFINITY_UTILS_H_
#define RDK_SERVICES_CPU_AFFINITY_AFFINITY_UTILS_H_

namespace rivermax
{
namespace dev_kit
{
namespace services
{
constexpr int INVALID_CORE_NUMBER = -1;

/**
* @brief: Sets CPU affinity for the rivermax internal thread.
*
* @param [in] cpu_affinity: CPU number to assign for the thread.
*
* @note: TODO: The return value will be changed to @ref ReturnStatus in the future.
*
* @return: true if the CPU affinity was set successfully, false otherwise.
*/
bool set_rivermax_thread_cpu_affinity(int cpu);
/**
* @brief: Sets CPU affinity for the current thread.
*
* @param [in] cpu: A CPU core number to assign to the current thread.
*/
void set_current_thread_affinity(const int cpu);

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_CPU_AFFINITY_AFFINITY_UTILS_H_ */
