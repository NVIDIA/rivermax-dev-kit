/*
 * Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
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
