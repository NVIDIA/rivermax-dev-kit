/*
 * Copyright © 2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef RMAX_APPS_LIB_LIB_CORE_CHUNK_RECEIVE_CHUNK_H_

#include <cstddef>
#include <string>

#include <rivermax_api.h>

#include "core/chunk/chunk_interface.h"

namespace ral
{
namespace lib
{
namespace core
{
/**
 * @brief: RX API chunk interface class.
 */
class ReceiveChunk : public IChunk
{
private:
    rmax_in_completion m_rmax_comp;
public:
    /**
     * @brief: ReceiveChunk default constructor.
     */
    ReceiveChunk();
    /**
     * @brief: Return number of packets in the chunk.
     *
     * @return: Number of packets.
     */
    virtual size_t get_length() const override { return m_rmax_comp.chunk_size; }
    /**
     * @brief: Returns Rivermax internal completion.
     *
     * @return: Pointer to the underlay Rivermax completion.
     */
    rmax_in_completion* get_completion() { return &m_rmax_comp; }
    /**
     * @brief: Returns Rivermax internal header pointer.
     *
     * @return: Pointer to the underlay Rivermax header array.
     */
    void* get_header_ptr() const { return m_rmax_comp.hdr_ptr; }
    /**
     * @brief: Returns Rivermax internal payload pointer.
     *
     * @return: Pointer to the underlay Rivermax payload array.
     */
    void* get_payload_ptr() const { return m_rmax_comp.data_ptr; }
};

} // namespace core
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_CORE_CHUNK_RECEIVE_CHUNK_H_
#endif /* RMAX_APPS_LIB_LIB_CORE_CHUNK_RECEIVE_CHUNK_H_ */
