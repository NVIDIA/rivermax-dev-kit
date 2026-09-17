/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef RDK_IO_NODE_IO_NODE_INTERFACE_H_
#define RDK_IO_NODE_IO_NODE_INTERFACE_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <type_traits>

#include "rdk/io_node/io_node_settings.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/error_handling/signal_handler.h"
#include "rdk/services/utils/lifecycle.h"

using namespace rdk::services;

namespace rdk
{
namespace io_node
{

/**
 * @brief: Interface and base for an I/O node.
 *
 * A thread context that owns one or more streams. Implements @ref ILifecycle.
 *
 * @tparam SettingsType: Settings type, deriving from @ref IONodeSettings. Held by value; use
 *                       @ref get_settings to read it.
 */
template<typename SettingsType>
class IIONode : public ILifecycle
{
    static_assert(std::is_base_of<IONodeSettings, SettingsType>::value,
        "SettingsType must inherit from IONodeSettings");
public:
    virtual ~IIONode() = default;
    /**
     * @brief: Runs the node as a thread context.
     *
     * Calls @ref start and discards its result, so a node can be passed directly to
     * @ref std::thread.
     */
    virtual void operator()() final { start(); }
    /**
     * @brief: Runs the node's worker.
     *
     * Sealed: runs @ref do_work, then unconditionally publishes completion.
     *
     * @return: Status the worker finished with.
     */
    ReturnStatus start() final
    {
        ReturnStatus rc = do_work();
        if (should_stop()) {
            rc = ReturnStatus::success;
        }
        publish_completion();
        return rc;
    }
    /**
     * @brief: Returns the settings the node was built from.
     *
     * @return: The node's settings.
     */
    const SettingsType& get_settings() const { return m_settings; }
    /**
     * @brief: Requests the node to stop.
     *
     * Non-blocking, idempotent, thread-safe. Does not join the worker or destroy its streams.
     */
    void stop() noexcept override
    {
        State expected = State::active;
        m_state.compare_exchange_strong(expected, State::stop_requested);
    }
    /**
     * @brief: Waits for the worker to publish its completion.
     *
     * Returns immediately if the worker has already published completion. Otherwise blocks for
     * @p timeout, then reports whether it has.
     *
     * @param [in] timeout: Maximum time to wait.
     *
     * @return: @ref ReturnStatus::success if the worker finished; @ref ReturnStatus::still_active
     *          if @p timeout elapsed first.
     */
    ReturnStatus await(std::chrono::nanoseconds timeout) const override
    {
        if (m_state.load() == State::stopped) {
            return ReturnStatus::success;
        }
        std::this_thread::sleep_for(timeout);
        return m_state.load() == State::stopped ?
            ReturnStatus::success : ReturnStatus::still_active;
    }

protected:
    explicit IIONode(const SettingsType& settings) : m_settings(settings) { validate_settings(); }
    explicit IIONode(SettingsType&& settings) : m_settings(std::move(settings)) { validate_settings(); }
    SettingsType m_settings;
    /**
     * @brief: Checks whether the worker should stop.
     *
     * True once @ref stop has been called on this node, or the process has received SIGINT.
     *
     * @return: true if the worker should stop.
     */
    bool should_stop() const noexcept
    {
        return m_state.load() != State::active || SignalHandler::get_received_signal() >= 0;
    }
    /**
     * @brief: Retries an operation while it reports a given status, unless a stop is requested.
     *
     * @param [in] op: Operation to retry. Called at least once.
     * @param [in] retry_on: Status that triggers a retry.
     *
     * @return: The last status returned by @p op.
     */
    template <typename Op>
    ReturnStatus retry_while(Op&& op, ReturnStatus retry_on)
    {
        ReturnStatus rc;
        do {
            rc = op();
        } while (rc == retry_on && !should_stop());
        return rc;
    }

private:
    /**
     * @brief: Rejects settings a node cannot be built from.
     */
    void validate_settings() const
    {
        if (m_settings.validate() != ReturnStatus::success) {
            throw std::runtime_error("Invalid I/O node settings");
        }
    }
    /**
     * @brief: The node's own work: setup, main loop, teardown.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus do_work() = 0;
    /**
     * @brief: Publishes the worker's completion.
     */
    void publish_completion() noexcept
    {
        m_state.store(State::stopped);
    }

    enum class State : uint8_t { active, stop_requested, stopped };

    std::atomic<State> m_state{State::active};
};

} // namespace io_node
} // namespace rdk

#endif /* RDK_IO_NODE_IO_NODE_INTERFACE_H_ */
