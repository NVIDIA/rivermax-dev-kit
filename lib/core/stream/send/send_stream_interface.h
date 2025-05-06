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

#ifndef RMAX_APPS_LIB_LIB_CORE_STREAM_SEND_SEND_STREAM_INTERFACE_H_

#include "core/stream/single_stream_interface.h"

namespace ral
{
namespace lib
{
namespace core
{
/**
 * @brief: Send stream interface.
 */
class ISendStream : public ISingleStream
{
protected:
    size_t m_num_of_chunks;
public:
    /**
     * @brief: ISendStream class constructor.
     *
     * @param [in] local_address: Network address of the NIC.
     */
    ISendStream(const TwoTupleFlow& local_address);
    virtual ~ISendStream() = default;
    std::ostream& print(std::ostream& out) const override;
    /**
     * @brief: Returns number of chunks used in the stream.
     *
     * @return: Number of chunks.
     */
    size_t get_num_of_chunks() { return m_num_of_chunks; }
};

} // namespace core
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_CORE_STREAM_SEND_SEND_STREAM_INTERFACE_H_
#endif /* RMAX_APPS_LIB_LIB_CORE_STREAM_SEND_SEND_STREAM_INTERFACE_H_ */
