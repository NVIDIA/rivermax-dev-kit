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

#ifndef RMAX_APPS_LIB_LIB_CORE_CHUNK_CHUNK_INTERFACE_H_

namespace ral
{
namespace lib
{
namespace core
{
constexpr size_t BLOCKING_COMMIT_RETRIES = 1000000;
constexpr size_t BLOCKING_CHUNK_RETRIES = 1000000;

/**
 * @brief: Chunk interface class.
 *
 * This interfaces indented to wrap Rivermax chunk.
 */
class IChunk
{
public:
    /**
     * @brief: IChunk default constructor.
     */
    IChunk() = default;
    virtual ~IChunk() = default;
    /**
     * @brief: Returns the length of the chunk.
     *
     * @returns: Chunk length in packets.
     */
    virtual size_t get_length() const = 0;
};

} // namespace core
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_CORE_CHUNK_CHUNK_INTERFACE_H_
#endif /* RMAX_APPS_LIB_LIB_CORE_CHUNK_CHUNK_INTERFACE_H_ */
