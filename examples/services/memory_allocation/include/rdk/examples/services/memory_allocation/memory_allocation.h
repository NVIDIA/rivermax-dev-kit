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

#ifndef RDK_EXAMPLES_SERVICES_MEMORY_ALLOCATION_MEMORY_ALLOCATION_H_
#define RDK_EXAMPLES_SERVICES_MEMORY_ALLOCATION_MEMORY_ALLOCATION_H_

#include "rdk/services/error_handling/return_status.h"
#include "rdk/examples/base_example.h"

using namespace rivermax::dev_kit::services;

namespace rivermax
{
namespace dev_kit
{
namespace examples
{
namespace services
{

/**
 * @brief: Memory allocation example.
 *
 * This example demonstrates the usage of memory allocators by a client application.
 * It showcases allocation using HugePage and GPU memory allocators.
 */
class MemoryAllocationExample : public BaseExample
{
public:
    /**
     * @brief: Constructor for the @ref MemoryAllocationExample class.
     */
    MemoryAllocationExample();

    ReturnStatus operator()() override;
    void add_cli_options() override;
};

} // namespace services
} // namespace examples
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_EXAMPLES_SERVICES_MEMORY_ALLOCATION_MEMORY_ALLOCATION_H_ */
