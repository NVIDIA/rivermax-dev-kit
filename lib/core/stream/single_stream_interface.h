/*
 * Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef RMAX_APPS_LIB_LIB_CORE_SINGLE_STREAM_STREAM_INTERFACE_H_

#include <cstddef>
#include <ostream>

#include <rivermax_api.h>

#include "core/flow/flow.h"
#include "core/stream/stream_interface.h"

namespace ral
{
namespace lib
{
namespace core
{
/**
 * @brief: Single stream interface (id and local NIC address).
 */
class ISingleStream : public IStream
{
protected:
    TwoTupleFlow m_local_address;
    rmx_stream_id m_stream_id;
    bool m_stream_id_set;
public:
    /**
     * @brief: ISingleStream class constructor.
     *
     * @param [in] local_address: Network address of the NIC.
     */
    ISingleStream(const TwoTupleFlow& local_address);
    virtual ~ISingleStream() = default;
    std::ostream& print(std::ostream& out) const override;
    /**
     * @brief: Returns Rivermax stream ID.
     *
     * @return: Stream ID.
     */
    rmx_stream_id get_id() const { return m_stream_id; }
};

} // namespace core
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_CORE_SINGLE_STREAM_STREAM_INTERFACE_H_
#endif /* RMAX_APPS_LIB_LIB_CORE_SINGLE_STREAM_STREAM_INTERFACE_H_ */
