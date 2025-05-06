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

#ifndef RDK_IO_NODE_RECEIVERS_RTP_EVENT_NOTIFIER_INTERFACE_H_
#define RDK_IO_NODE_RECEIVERS_RTP_EVENT_NOTIFIER_INTERFACE_H_

#include <cstdint>

namespace rivermax
{
namespace dev_kit
{
namespace io_node
{
/**
 * @brief: RTP event notifier interface.
 */
class IRTPEventHandler
{
public:
    virtual ~IRTPEventHandler() = default;
    /**
     * @brief: Notifies an observer about a new event of RTP stream.
     *
     * This function will be invoked on the detection of the event on the stream.
     *
     * @param [in] rtp_timestamp: RTP timestamp.
     * @param [in] receiver_timestamp: Receiver's clock timestamp at the time of the event.
     */
    virtual void notify_rtp_event(uint32_t rtp_timestamp, uint64_t receiver_timestamp) = 0;
};

} // namespace io_node
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_IO_NODE_RECEIVERS_RTP_EVENT_NOTIFIER_INTERFACE_H_ */
