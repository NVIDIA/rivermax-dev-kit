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

#ifndef RDK_SERVICES_STATISTICS_STATISTICS_READER_H_
#define RDK_SERVICES_STATISTICS_STATISTICS_READER_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <thread>
#include <unordered_map>

#include <rmx_stats_api.h>

#include "rdk/services/error_handling/error_handling.h"
#include "rdk/services/utils/lifecycle.h"


namespace rdk
{
namespace services
{

class StatisticsReader;
typedef std::function<void(StatisticsReader *reader)> statistics_handler_cb_t;
typedef std::unordered_map<rmx_stats_type, statistics_handler_cb_t> statistics_handler_cb_map_t;

/**
 * @brief: Basic class for Rivermax Dev Kit statistics reader.
 *
 * This is a basic class offers common statistics reader implementation.
 * The user of this interface can override it's virtual methods.
 */
class StatisticsReader : public ILifecycle
{
protected:
    /** Statistics related */
    /* A descriptor for statistics consumer configuration */
    rmx_stats_config m_stats_config;
    /* A descriptor for a statistics message */
    rmx_stats_message m_stats_message;
    /* A descriptor for statistics consumer */
    rmx_stats_consumer m_stats_consumer;
    /* Statistics generating process ID, default 0 is local process */
    uint32_t m_process_id;
    /* Indicator on whether statistics consumer created correctly */
    bool m_stats_consumer_created;
    /** Statistics filter related parameters  */
    /* If not equal 0 - filter statistics by by m_thread_id */
    uint32_t m_thread_id;
    /* If not equal UINT_MAX - filter statistics by session_id */
    uint32_t m_session_id;
    /** CPU affinity related */
    int m_cpu_core_affinity;
    /** Reader related */
    /* Sleep duration in seconds before every cycle of statistics reading */
    uint32_t m_update_time_sec;
    /* Statistics type to handler method map */
    statistics_handler_cb_map_t m_statistics_handler_cb_map;
    enum class State : uint8_t { active, stop_requested, stopped };
    std::atomic<State> m_state{State::active};
public:
    StatisticsReader();
    virtual ~StatisticsReader();
    /**
     * @brief: Initializes statistics discriptors and common default settings.
     *
     * This method creates statistics consumer and attach to statistics message queue.
     *
     * @return: Status of the operation as defined by @ref ReturnStatus.
     */
    virtual ReturnStatus init_stats_reader();
    /**
     * @brief: Sets statistics configuration.
     *
     * The user can override this function, to override specific default statistics settings
     * like the types of the statistics messages Rivermax is required to generate.
     *
     * @return: Status of the operation as defined by @ref ReturnStatus.
     */
    virtual ReturnStatus set_stats_config();
    /**
     * @brief: Runs the reader as a thread context.
     *
     * Calls @ref start and nothing else, so the reader can be passed straight to
     * @ref std::thread. Sealed, like @ref start.
     */
    virtual void operator()() final { start(); }
    /**
     * @brief: Runs the reader's worker.
     *
     * Sealed: initializes the statistics reader and reads statistics messages in a loop until
     * asked to stop, then publishes completion.
     *
     * @return: Status the reader finished with.
     */
    ReturnStatus start() final;
    /**
     * @brief: Requests the statistics reader to stop.
     *
     * Non-blocking, idempotent, thread-safe. The reader observes the request at the end of
     * its current update cycle, in the same way it observes a process signal.
     */
    void stop() noexcept override
    {
        State expected = State::active;
        m_state.compare_exchange_strong(expected, State::stop_requested);
    }
    /**
     * @brief: Waits for the reader to publish its completion.
     *
     * Returns immediately if the reader has already published completion. Otherwise blocks
     * for @p timeout, then reports whether it has.
     *
     * @param [in] timeout: Maximum time to wait.
     *
     * @return: @ref ReturnStatus::success if the reader finished; @ref ReturnStatus::still_active
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
    /**
     * @brief: Sets process ID.
     *
     * If process ID value is set to value different from 0, attach to statistics generated by remote process.
     * If the process do not use @ref rivermax_init it should call @ref rmx_stats_init
     * before starting using statistics related operations.
     *
     * @param [in] process_id: Process ID.
     */
    void set_sprocess_id(uint32_t process_id) { m_process_id = process_id; }
    /**
     * @brief: Sets filtered session ID.
     *
     * This is used by the skip method to skip session IDs, if this value different from UINT_MAX.
     *
     * @param [in] session_id: Session ID.
     */
    void set_session_id(uint32_t session_id) { m_session_id = session_id; }
    /**
     * @brief: Sets filtered thread ID.
     *
     * This is used by the skip method to skip thread IDs, if this value different from 0.
     *
     * @param [in] thread_id: Thread ID.
     */
    void set_thread_id(uint32_t thread_id) { m_thread_id = thread_id; }
    /**
     * @brief: Sets CPU core affinity for statistics reader thread.
     *
     * @param [in] cpu_core_affinity: Core number.
     */
    void  set_cpu_core_affinity(int cpu_core_affinity) { m_cpu_core_affinity = cpu_core_affinity; }
protected:
    /**
     * @brief: Checks whether the reader should stop.
     *
     * True once @ref stop has been called on this reader, or the process has received SIGINT.
     *
     * @return: true if the reader should stop.
     */
    bool should_stop() const noexcept
    {
        return m_state.load() != State::active || SignalHandler::get_received_signal() >= 0;
    }
    /**
     * @brief: Reads one statistic message in to m_stats_message data member.
     *
     * @return: Status of the operation:
     *          @ref ReturnStatus::success - There might be additional messages to be read.
     *          @ref ReturnStatus::no_free_chunks - No additional messages to be read.
     *          @ref ReturnStatus::signal_received - If operation was interrupted by an OS signal.
     */
    virtual ReturnStatus read_stats_msg();
    /**
     * @brief: Checks is the message should be skiped.
     *
     * @return: true if the message should be skiped.
     */
    virtual bool skip();
    /**
     * @brief: Handles the statistic message stored in m_stats_message data member.
     *
     *  The curent implementation of handling the message is to write it to the standard output.
     *  The user of this interface can override statistics type specific functions to handle the message differently.
     */
    virtual void handle_msg();
    /**
     * @brief: Handles the statistic message type session start.
     */
    virtual void handle_session_start();
    /**
     * @brief: Handles the statistic message type session stop.
     */
    virtual void handle_session_stop();
    /**
     * @brief: Handles the statistic message type session runtime.
     */
    virtual void handle_session_run();
    /**
     * @brief: Handles the statistic message type TX queue.
     */
    virtual void handle_tx_queue();
    /**
     * @brief: Handles the statistic message type RX queue.
     */
    virtual void handle_rx_queue();
    /**
     * @brief: Handles statistic message with unknown type..
     */
    virtual void handle_unknown();
    /**
     * @brief: Displasy message header.
     */
    virtual void display_hdr();
private:
    /**
     * @brief: The reader's own work.
     *
     * Called by @ref start, which publishes completion unconditionally once this returns.
     *
     * @return: Status of the operation.
     */
    ReturnStatus do_work();
    /**
     * @brief: Publishes the reader's completion.
     */
    void publish_completion() noexcept { m_state.store(State::stopped); }
    static statistics_handler_cb_map_t s_statistics_handler_cb_map;
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_STATISTICS_STATISTICS_READER_H_ */
