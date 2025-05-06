/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef RDK_CORE_CHUNK_CHUNK_INTERFACE_H_
#define RDK_CORE_CHUNK_CHUNK_INTERFACE_H_

#include <cstddef>

namespace rivermax
{
namespace dev_kit
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
    virtual ~IChunk() = default;
    /**
     * @brief: Returns the length of the chunk.
     *
     * @returns: Chunk length in packets.
     */
    virtual size_t get_length() const = 0;
protected:
    /**
     * @brief: IChunk default constructor.
     */
    IChunk() = default;
};

/**
 * @brief: RX API chunk interface class.
 */
class IReceiveChunk : public IChunk
{
public:
    virtual ~IReceiveChunk() = default;

    /**
     * @brief: Returns Rivermax internal header pointer.
     *
     * @return: Pointer to the underlay Rivermax header array.
     */
    virtual const void* get_header_ptr() const = 0;
    /**
     * @brief: Returns Rivermax internal payload pointer.
     *
     * @return: Pointer to the underlay Rivermax payload array.
     */
    virtual const void* get_payload_ptr() const = 0;
    /**
     * @brief: Returns status of Header-Data-Split mode.
     *
     * @return: true if Header-Data-Split mode is enabled.
     */
    virtual bool is_header_data_split_on() const = 0;
protected:
    /**
     * @brief: IReceiveChunk default constructor.
     */
    IReceiveChunk() = default;
};

} // namespace core
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_CORE_CHUNK_CHUNK_INTERFACE_H_ */
