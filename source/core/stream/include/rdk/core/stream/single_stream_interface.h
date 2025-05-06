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

#ifndef RDK_CORE_STREAM_SINGLE_STREAM_STREAM_INTERFACE_H_
#define RDK_CORE_STREAM_SINGLE_STREAM_STREAM_INTERFACE_H_

#include <ostream>

#include <rivermax_api.h>

#include "rdk/core/flow/flow.h"
#include "rdk/core/stream/stream_interface.h"

namespace rivermax
{
namespace dev_kit
{
namespace core
{
/**
 * @brief: Single stream interface.
 */
class ISingleStream : public IStream
{
protected:
    TwoTupleFlow m_local_address;
public:
    virtual ~ISingleStream() = default;
    std::ostream& print(std::ostream& out) const override;
protected:
    /**
     * @brief: ISingleStream class constructor.
     *
     * @param [in] local_address: Network address of the NIC.
     */
    ISingleStream(const TwoTupleFlow& local_address);
};

} // namespace core
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_CORE_STREAM_SINGLE_STREAM_STREAM_INTERFACE_H_ */
