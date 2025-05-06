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

#ifndef RDK_APPS_MEMORY_STRATEGY_INTERFACE_H_
#define RDK_APPS_MEMORY_STRATEGY_INTERFACE_H_

#include "rdk/services/error_handling/return_status.h"

using namespace rivermax::dev_kit::services;

namespace rivermax
{
namespace dev_kit
{
namespace apps
{

/**
 * @brief: Interface for memory strategy.
 *
 * This interface defines the methods required for managing memory layout,
 * allocation, registration, and application for memory components.
 */
class IMemoryStrategy
{
public:
    virtual ~IMemoryStrategy() = default;
    /**
     * @brief: Determines the memory layout for all components.
     *
     * This function queries each memory component for its memory layout,
     * calculates the total memory required, and stores the memory layout.
     *
     * @return: Returns status of the operation.
     */
    virtual ReturnStatus determine_memory_layout() = 0;
    /**
     * @brief: Applies the memory layout to all components.
     *
     * This method applies the provided memory layout to all components.
     *
     * @return: Returns status of the operation.
     */
    virtual ReturnStatus apply_memory_layout() = 0;
    /**
     * @brief: Allocates memory for all components.
     *
     * This method is responsible for allocation of the required memory for
     * all components in the application.
     *
     * @return: Returns status of the operation.
     */
    virtual ReturnStatus allocate_memory() = 0;
    /**
     * @brief: Registers previously allocated memory for all components if requested.
     *
     * @return: Returns status of the operation.
     */
     virtual ReturnStatus register_memory() = 0;
    /**
     * @brief: Unregisters previously registered memory for all components.
     */
    virtual void unregister_memory() = 0;
protected:
    /**
     * @brief: Default constructor.
     */
    IMemoryStrategy() = default;
};

} // namespace apps
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_APPS_MEMORY_STRATEGY_INTERFACE_H_ */
