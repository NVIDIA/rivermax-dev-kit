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

#ifndef RMAX_APPS_LIB_LIB_CORE_CHUNK_GENERIC_CHUNK_H_

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
 * @brief: Generic API chunk interface class.
 *
 * This interfaces intended to wrap Rivermax Generic API chunk.
 */
class GenericChunk : public IChunk
{
private:
    rmax_chunk* m_rmax_chunk;
public:
    /**
     * @brief: GenericChunk default constructor.
     */
    GenericChunk();
    /**
     * @brief: GenericChunk constructor.
     *
     * @param [in] _rmax_chunk: Rivermax generic API chunk.
     */
    GenericChunk(rmax_chunk* _rmax_chunk);
    virtual size_t get_length() const override { return m_rmax_chunk->size; };
    /**
     * @brief: Returns Rivermax internal chunk.
     *
     * @returns: Pointer to the underlay Rivermax chunk.
     */
    rmax_chunk* get_rmax_chunk() const { return m_rmax_chunk; }
    /**
     * @brief: Returns Rivermax internal packets pointer.
     *
     * @returns: Pointer to the underlay Rivermax packets array.
     */
    rmax_packet* get_packets_ptr() const { return m_rmax_chunk->packets; };
    /**
     * @brief: Returns Rivermax internal chunk context.
     *
     * @returns: Pointer to the underlay Rivermax chunk context.
     */
    void* get_context() const { return m_rmax_chunk->chunk_ctx; };
};

} // namespace core
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_CORE_CHUNK_GENERIC_CHUNK_H_
#endif /* RMAX_APPS_LIB_LIB_CORE_CHUNK_GENERIC_CHUNK_H_ */
