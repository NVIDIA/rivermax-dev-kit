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

#ifndef RDK_CORE_STREAM_RECEIVE_AGGREGATE_RECEIVE_STREAM_INTERFACE_H_
#define RDK_CORE_STREAM_RECEIVE_AGGREGATE_RECEIVE_STREAM_INTERFACE_H_

#include <ostream>

#include "rdk/core/stream/receive/receive_stream_interface.h"

namespace rivermax
{
namespace dev_kit
{
namespace core
{
/**
 * @brief: Aggregate receive stream interface.
 */
class IAggregateReceiveStream : public IReceiveStream
{
public:
    virtual ~IAggregateReceiveStream() = default;
    std::ostream& print(std::ostream& out) const override;
    std::string get_stream_name() const override;

protected:
    /**
     * @brief: IAggregateStream class constructor.
     */
    IAggregateReceiveStream();
};

} // namespace core
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_CORE_STREAM_RECEIVE_AGGREGATE_RECEIVE_STREAM_INTERFACE_H_ */
