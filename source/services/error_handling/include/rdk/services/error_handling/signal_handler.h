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

#ifndef RDK_SERVICES_ERROR_HANDLING_SIGNAL_HANDLER_H_
#define RDK_SERVICES_ERROR_HANDLING_SIGNAL_HANDLER_H_

#include <atomic>
#include <unordered_map>
#include <csignal>
#include <functional>

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Signal handler callback definition.
 */
typedef std::function<void(int)> signal_handler_t;
typedef std::unordered_map<int, signal_handler_t> signal_handlers_map_t;
/**
 * @brief: Signal handler class.
 */
class SignalHandler
{
public:
    static std::atomic_int s_received_signal;
    static signal_handlers_map_t s_signal_handlers_map;
public:
    /**
     * @brief: SignalHandler constructor.
     *
     * @param [in] register_default_handler: Determines whether to register default signal handler for SIGINT signal.
     *                                       Defaults to false.
     * @note: The default handler for SIGINT signal is @ref SignalHandler::default_signal_handler.
     */
    SignalHandler(bool register_default_handler = false);
    /**
     * @brief: Returns received signal number.
     *
     * This method will return the received signal number atomically. It is thread safe.
     * In case no signal received, the returned value will be INT_MIN.
     *
     * @returns: Received signal number.
     */
    static inline int get_received_signal() { return SignalHandler::s_received_signal.load(); }
    /**
     * @brief: Registers signal handler callback.
     *
     * @param [in] signal: Signal number.
     * @param [in] handler: Signal handler callback function.
     */
    void register_signal_handler_callback(int signal, signal_handler_t handler);
private:
    /**
     * @brief: Initializes default signal handlers.
     *
     * This method uses @ref SignalHandler::default_signal_handler
     * as the handler. Currently, SIGINT is the only signal initialized.
     */
    void initialize_default_signal_handlers();
    /**
     * @brief: Default signal handler callback.
     *
     * The behavior of the method is to log the received signal number.
     *
     * @param [in] signal: Signal number.
     */
    static void default_signal_handler(int signal);
    /**
     * @brief: Signal handler arbiter.
     *
     * The method will chose the appropriate signal handler function
     * for the given signal number and invoke it.
     *
     * @param [in] signal: Signal number.
     */
    static void signal_handler_arbiter(int signal);
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_ERROR_HANDLING_SIGNAL_HANDLER_H_ */
