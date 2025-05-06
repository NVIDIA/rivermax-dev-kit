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

#ifndef RDK_SERVICES_ERROR_HANDLING_RETURN_STATUS_H_
#define RDK_SERVICES_ERROR_HANDLING_RETURN_STATUS_H_

namespace rivermax
{
namespace dev_kit
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
    frame_send_queue_full,
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_ERROR_HANDLING_RETURN_STATUS_H_ */
