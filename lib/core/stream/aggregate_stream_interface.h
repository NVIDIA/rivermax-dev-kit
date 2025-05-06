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

#ifndef RMAX_APPS_LIB_LIB_CORE_AGGREGATE_STREAM_STREAM_INTERFACE_H_

#include <cstddef>
#include <ostream>

#include "core/stream/stream_interface.h"

namespace ral
{
namespace lib
{
namespace core
{
/**
 * @brief: Aggregate stream interface (externally passed ID).
 */
class IAggregateStream : public IStream
{
protected:
    const size_t m_stream_id;
public:
    /**
     * @brief: IAggregateStream class constructor.
     *
     * @param [in] id: Aggregate stream ID.
     */
    IAggregateStream(size_t id);
    virtual ~IAggregateStream() = default;
    std::ostream& print(std::ostream& out) const override;
    /**
     * @brief: Returns stream ID.
     *
     * @return: Stream ID.
     */
    size_t get_id() const { return m_stream_id; }
};

} // namespace core
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_CORE_AGGREGATE_STREAM_STREAM_INTERFACE_H_
#endif /* RMAX_APPS_LIB_LIB_CORE_AGGREGATE_STREAM_STREAM_INTERFACE_H_ */
