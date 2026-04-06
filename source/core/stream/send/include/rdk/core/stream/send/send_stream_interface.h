/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef RDK_CORE_STREAM_SEND_SEND_STREAM_INTERFACE_H_
#define RDK_CORE_STREAM_SEND_SEND_STREAM_INTERFACE_H_

#include <ostream>
#include <vector>

#include "rdk/core/stream/stream_interface.h"
#include "rdk/core/flow/flow.h"

namespace rdk
{
namespace core
{
/**
 * @brief: Send stream interface.
 */
class ISendStream : public IStream
{
protected:
    std::vector<TwoTupleFlow> m_local_addresses;
public:
    virtual ~ISendStream() = default;
    std::ostream& print(std::ostream& out) const override;
protected:
    /**
     * @brief: ISendStream class constructor.
     *
     * @param [in] local_addresses: Network addresses of the NICs.
     */
    ISendStream(const std::vector<TwoTupleFlow>& local_addresses);
};

} // namespace core
} // namespace rdk

#endif /* RDK_CORE_STREAM_SEND_SEND_STREAM_INTERFACE_H_ */
