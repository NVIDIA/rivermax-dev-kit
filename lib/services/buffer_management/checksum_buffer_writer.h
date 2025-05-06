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

#ifndef RMAX_APPS_LIB_LIB_SERVICES_BUFFER_MANAGEMENT_CHECKSUME_BUFFER_WRITER_H_

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
 * Packet header portion that can be used to perform sequence (for dropped packets) and checksum checking.
 */
struct ChecksumHeader
{
    uint32_t sequence;
    uint32_t checksum;
};
/**
 * @brief: Writes data buffers that include checksum information.
 *
 * This class implements @ref ral::lib::services::IBufferWriter interface.
 * It fills the data buffer with random data, and the ChecksumHeader header
 * with a checksum corresponding to the data that was written.
 */
class ChecksumBufferWriter : public IBufferWriter
{
public:
    void write_buffer(IChunk& chunk, std::shared_ptr<MemoryUtils> mem_utils) override;
};

} // namespace services
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_SERVICES_BUFFER_MANAGEMENT_CHECKSUME_BUFFER_WRITER_H_
#endif /* RMAX_APPS_LIB_LIB_SERVICES_BUFFER_MANAGEMENT_CHECKSUME_BUFFER_WRITER_H_ */
