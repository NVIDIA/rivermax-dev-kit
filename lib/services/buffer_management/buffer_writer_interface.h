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
