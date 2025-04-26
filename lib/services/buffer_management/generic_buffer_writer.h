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

#ifndef RMAX_APPS_LIB_LIB_SERVICES_BUFFER_MANAGEMENT_GENERIC_BUFFER_WRITER_H_

#include "core/chunk/chunk.h"
#include "services/buffer_management/buffer_writer_interface.h"

using namespace ral::lib::core;

namespace ral
{
namespace lib
{
namespace services
{

/**
 * @brief: Generic data buffers writer.
 *
 * This class implements @ref ral::lib::services::IBufferWriter interface.
 * It writes the buffer by some internal logic.
 * There is no special reason why this buffer writer writes it's data the way it does.
 */
class GenericBufferWriter : public IBufferWriter
{
public:
    void write_buffer(IChunk& chunk, std::shared_ptr<MemoryUtils> mem_utils) override;
};

} // namespace services
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_SERVICES_BUFFER_MANAGEMENT_GENERIC_BUFFER_WRITER_H_
#endif /* RMAX_APPS_LIB_LIB_SERVICES_BUFFER_MANAGEMENT_GENERIC_BUFFER_WRITER_H_ */
