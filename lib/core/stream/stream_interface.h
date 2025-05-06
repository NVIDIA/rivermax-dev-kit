/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef RMAX_APPS_LIB_LIB_CORE_STREAM_STREAM_INTERFACE_H_

#include <cstddef>
#include <ostream>
#include <vector>
#include <functional>
#include <algorithm>

#include "services/error_handling/return_status.h"

namespace ral
{
namespace lib
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
public:
    /**
     * @brief: IStream class constructor.
     */
    IStream() = default;
    virtual ~IStream() = default;
    /**
     * @brief: Prints stream's parameters to a output stream.
     *
     * The method prints the parameters of the stream to be shown to the user to a output stream.
     * Implementors of @ref ral::lib::core::IStream can extend this operation by overriding this method
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
     * @brief: Overrides operator << for @ref ral::lib::core::IStream reference.
     */
    friend std::ostream& operator<<(std::ostream& out, IStream& stream)
    {
        stream.print(out);
        return out;
    }
    /**
     * @brief: Overrides operator << for @ref ral::lib::core::IStream pointer.
     */
    friend std::ostream& operator<<(std::ostream& out, IStream* stream)
    {
        stream->print(out);
        return out;
    }
    /**
     * @brief: Creates the stream.
     *
     * This method is responsible to create Rivermax stream.
     *
     * @return: Status of the operation.
     */
    virtual ral::lib::services::ReturnStatus create_stream() = 0;
    /**
     * @brief: Destroys the stream.
     *
     * This method is responsible to destroy Rivermax stream.
     *
     * @return: Status of the operation.
     */
    virtual ral::lib::services::ReturnStatus destroy_stream() = 0;
};

} // namespace core
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_CORE_STREAM_STREAM_INTERFACE_H_
#endif /* RMAX_APPS_LIB_LIB_CORE_STREAM_STREAM_INTERFACE_H_ */
