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

#ifndef RDK_CORE_MEMORY_LAYOUT_MEMORY_LAYOUT_INTERFACE_H_
#define RDK_CORE_MEMORY_LAYOUT_MEMORY_LAYOUT_INTERFACE_H_

#include <memory>

#include "rdk/services/error_handling/return_status.h"

using namespace rivermax::dev_kit::services;

namespace rivermax
{
namespace dev_kit
{
namespace core
{

/**
 * @brief: Base struct for memory layout request.
 *
 * This struct serves as a base for different memory layout configuration requests.
 */
struct MemoryLayoutRequest
{
public:
    virtual ~MemoryLayoutRequest() = default;
protected:
    MemoryLayoutRequest() = default;
};
/**
 * @brief: Base struct for memory layout response.
 *
 * This struct serves as a base for different memory layout configuration responses.
 */
struct MemoryLayoutResponse
{
public:
    virtual ~MemoryLayoutResponse() = default;
protected:
    MemoryLayoutResponse() = default;
};
/**
 * @brief: Interface for memory layout components.
 *
 * This interface defines the methods that memory layout components must implement.
 */
template <typename RequestType, typename ResponseType>
class IMemoryLayoutComponent
{
public:
    virtual ~IMemoryLayoutComponent() = default;
    /**
     * @brief: Initializes memory layout.
     *
     * This method initializes the memory layout for the component.
     * It should called before any other method.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus initialize_memory_layout() = 0;
    /**
     * @brief: Determines the memory layout.
     *
     * This method determines the memory layout for the component.
     * At the stage of determining the memory layout, the memory layout should be initialized.
     *
     * @param [out] memory_layout_request: Memory layout object to be determined.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus determine_memory_layout(RequestType& memory_layout_request) const = 0;
    /**
     * @brief: Applies the memory layout.
     *
     * This method applies the provided memory layout to the component.
     *
     * @param [in] memory_layout_response: Memory layout object to be applied.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus apply_memory_layout(const ResponseType& memory_layout_response) = 0;
    /**
     * @brief: Validates the memory layout.
     *
     * This method validates the provided memory layout.
     *
     * @param [in] memory_layout_response: Memory layout object to be validated.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus validate_memory_layout(const ResponseType& memory_layout_response) const = 0;
protected:
    /**
     * @brief: Constructor for IMemoryLayoutComponent.
     */
    IMemoryLayoutComponent() = default;
};

/**
 * @brief: Wrapper class for memory layout components.
 *
 * This class provides a type-erased wrapper for memory layout components,
 * allowing different types of memory layout components to be stored in the same container.
 */
class MemoryLayoutComponentWrapper
{
public:
    /**
     * @brief: Constructor for MemoryLayoutComponentWrapper.
     *
     * @param [in] component: Shared pointer to the memory layout component.
     */
    template <typename ComponentType>
    MemoryLayoutComponentWrapper(const std::shared_ptr<ComponentType>& component) : m_component(component) {}
    /**
     * @brief: Determines the memory layout.
     *
     * This method determines the memory layout for the component.
     * At the stage of determining the memory layout, the memory layout should be initialized.
     *
     * @param [out] memory_layout_request: Memory layout object to be determined.
     *
     * @return: Status of the operation.
     */
    ReturnStatus determine_memory_layout(MemoryLayoutRequest& memory_layout_request) const
    {
        return m_component->determine_memory_layout(memory_layout_request);
    }
    /**
     * @brief: Applies the memory layout.
     *
     * This method applies the provided memory layout to the component.
     *
     * @param [in] memory_layout_response: Memory layout object to be applied.
     *
     * @return: Status of the operation.
     */
    ReturnStatus apply_memory_layout(const MemoryLayoutResponse& memory_layout_response)
    {
        return m_component->apply_memory_layout(memory_layout_response);
    }
    /**
     * @brief: Validates the memory layout.
     *
     * This method validates the provided memory layout.
     *
     * @param [in] memory_layout_response: Memory layout object to be validated.
     *
     * @return: Status of the operation.
     */
    ReturnStatus validate_memory_layout(const MemoryLayoutResponse& memory_layout_response) const
    {
        return m_component->validate_memory_layout(memory_layout_response);
    }
private:
    std::shared_ptr<IMemoryLayoutComponent<MemoryLayoutRequest, MemoryLayoutResponse>> m_component;
};

} // namespace core
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_CORE_MEMORY_LAYOUT_MEMORY_LAYOUT_INTERFACE_H_ */
