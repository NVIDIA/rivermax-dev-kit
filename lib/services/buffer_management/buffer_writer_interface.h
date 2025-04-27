/*
 * Copyright © 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef RMAX_APPS_LIB_LIB_SERVICES_BUFFER_MANAGEMENT_BUFFER_WRITER_INTERFACE_H_

#include "core/chunk/chunk.h"
#include "services/memory_management/memory_management.h"

using namespace ral::lib::core;

namespace ral
{
namespace lib
{
namespace services
{

/**
 * @brief: Data buffers writer interface.
 */
class IBufferWriter
{
public:
    /**
     * @brief: IBufferWriter default constructor.
     */
    IBufferWriter() = default;
    virtual ~IBufferWriter() = default;
    /**
     * @brief: Writes data buffers.
     *
     * Write the data to the given chunk.
     *
     * @param [in] chunk: Chunk reference.
     * @param [in] mem_utils: Memory utilities.
     */
    virtual void write_buffer(IChunk& chunk, std::shared_ptr<MemoryUtils> mem_utils) = 0;
};

} // namespace services
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_SERVICES_BUFFER_MANAGEMENT_BUFFER_WRITER_INTERFACE_H_
#endif /* RMAX_APPS_LIB_LIB_SERVICES_BUFFER_MANAGEMENT_BUFFER_WRITER_INTERFACE_H_ */
