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

#ifndef RDK_CORE_STREAM_STREAM_INTERFACE_H_
#define RDK_CORE_STREAM_STREAM_INTERFACE_H_

#include <ostream>
#include <functional>
#include <algorithm>

#include <rivermax_api.h>

#include "rdk/services/error_handling/return_status.h"

using namespace rivermax::dev_kit::services;

namespace rivermax
{
namespace dev_kit
{
namespace core
{

/**
 * @brief: Stream settings interface.
 */
template <typename StreamType, typename DescriptorType>
class IStreamSettings
{
public:
    /**
    * @brief: Function type to build a logical part of stream descriptor.
    */
    using StreamParamSetter = std::function<void(StreamType* settings, DescriptorType& descr)>;
    /**
    * @brief: Sequence of @ref StreamParamSetter functions to invoke when building a stream descriptor.
    */
    using SetterSequence = std::initializer_list<StreamParamSetter>;
    /**
     * @brief: IStreamSettings class constructor.
     */
    IStreamSettings(const SetterSequence& build_steps) : m_build_steps(build_steps) {}
    virtual ~IStreamSettings() = default;
    /**
     * @brief: Stream opaque structure builder function.
     *
     * @param [in] descr: Pointer to opaque stream descriptor structure.
     */
    void build(StreamType& object, DescriptorType& descr)
    {
        std::for_each(m_build_steps.begin(), m_build_steps.end(),
            [&](StreamParamSetter step) { step(&object, descr); });
    }
protected:
    /**
     * @brief: Functions to build logical parts of stream descriptor.
     */
    const SetterSequence& m_build_steps;
};

/**
 * @brief: Stream interface.
 */
class IStream
{
protected:
    bool m_stream_created = false;
    rmx_stream_id m_stream_id = 0;
public:
    virtual ~IStream() = default;
    /**
     * @brief: Prints stream's parameters to a output stream.
     *
     * The method prints the parameters of the stream to be shown to the user to a output stream.
     * Implementors of @ref IStream can extend this operation by overriding this method
     * in order to add the derived stream parameters.
     *
     * @param [out] out: Output stream parameter print to.
     *
     * @note: In case this method overridden, the implementor should remember to call
              the base print method, if he wants the base parameters to be printed as well.
     *
     * @return: Output stream.
     */
    virtual std::ostream& print(std::ostream& out) const { return out; }
    /**
     * @brief: Overrides operator << for @ref IStream reference.
     */
    friend std::ostream& operator<<(std::ostream& out, IStream& stream)
    {
        stream.print(out);
        return out;
    }
    /**
     * @brief: Overrides operator << for @ref IStream pointer.
     */
    friend std::ostream& operator<<(std::ostream& out, IStream* stream)
    {
        stream->print(out);
        return out;
    }
    /**
     * @brief: Returns stream ID.
     *
     * @return: Stream ID.
     */
    virtual rmx_stream_id get_id() const { return m_stream_id; }
    /**
     * @brief: Creates the stream.
     *
     * This method is responsible to create Rivermax stream.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus create_stream() = 0;
    /**
     * @brief: Destroys the stream.
     *
     * This method is responsible to destroy Rivermax stream.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus destroy_stream() = 0;
protected:
    /**
     * @brief: IStream class constructor.
     */
    IStream() = default;
};

} // namespace core
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_CORE_STREAM_STREAM_INTERFACE_H_ */
