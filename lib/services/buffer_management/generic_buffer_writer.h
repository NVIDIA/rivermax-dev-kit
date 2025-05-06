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
