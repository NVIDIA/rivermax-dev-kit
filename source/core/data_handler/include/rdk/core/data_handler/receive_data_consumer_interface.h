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

#ifndef RDK_CORE_DATA_HANDLER_RECEIVE_DATA_CONSUMER_INTERFACE_H_
#define RDK_CORE_DATA_HANDLER_RECEIVE_DATA_CONSUMER_INTERFACE_H_

#include <cstddef>

#include "rdk/core/chunk/chunk.h"
#include "rdk/core/stream/stream.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
{
namespace core
{
/**
 * @brief: Interface for consuming received data packets.
 *
 * Custom Data Consumption Interface:
 * IReceiveDataConsumer interface, allows users to implement custom logic
 * for consuming received data packets. This enables flexible data handling per stream
 * through user-defined implementations.
 */
class IReceiveDataConsumer
{
public:
    virtual ~IReceiveDataConsumer() = default;
    /**
    * @brief: Consumes packets from a given chunk.
    *
    * This function consumes the packets contained in the provided chunk and sets
    * the number of consumed packets.
    *
    * @param [in] chunk: Reference to the ReceiveChunk containing the packets.
    * @param [in] stream: Reference to the IReceiveStream associated with the chunk.
    * @param [out] consumed_packets: Reference to the number of consumed packets.
    *
    * @return: Status of the operation.
    */
    virtual ReturnStatus consume_chunk(
        const ReceiveChunk& chunk, const IReceiveStream& stream, size_t& consumed_packets) = 0;
};

/**
 * @brief: Default implementation of the IReceiveDataConsumer interface.
 *
 * Default Data Consumer Implementation:
 * Provides a default implementation of the IReceiveDataConsumer interface that
 * consumes all packets in the chunk without any custom processing. This can be
 * used as a placeholder or base class for more complex data consumers.
 */
class NullReceiveDataConsumer : public IReceiveDataConsumer
{
public:
    virtual ~NullReceiveDataConsumer() = default;
    ReturnStatus consume_chunk(
        const ReceiveChunk& chunk, const IReceiveStream& stream, size_t& consumed_packets) override {
        consumed_packets = chunk.get_length();
        return ReturnStatus::success;
    };
};

} // namespace core
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_CORE_DATA_HANDLER_RECEIVE_DATA_CONSUMER_INTERFACE_H_ */
