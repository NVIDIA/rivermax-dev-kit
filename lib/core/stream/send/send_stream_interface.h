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

#ifndef RMAX_APPS_LIB_LIB_CORE_STREAM_SEND_SEND_STREAM_INTERFACE_H_

#include "core/stream/single_stream_interface.h"

namespace ral
{
namespace lib
{
namespace core
{
/**
 * @brief: Number of retries for commit related blocking operations.
 */
constexpr size_t BLOCKING_COMMIT_RETRIES = 1000000;
/**
 * @brief: Number of retries for chunk related blocking operations.
 */
constexpr size_t BLOCKING_CHUNK_RETRIES = 1000000;
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
    /**
     * @brief: Prints stream's parameters to a output stream.
     *
     * The method prints the parameters of the stream to be shown to the user to a output stream.
     *
     * @param [out] out: Output stream parameter print to.
     *
     * @return: Output stream.
     */
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
