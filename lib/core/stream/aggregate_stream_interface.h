/*
 * Copyright © 2017-2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
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
