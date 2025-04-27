/*
 * Copyright © 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef RMAX_APPS_LIB_LIB_SERVICES_ERROR_HANDLING_SIGNAL_HANDLER_H_

#include <atomic>
#include <unordered_map>
#include <csignal>
#include <functional>

namespace ral
{
namespace lib
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
     * @param [in] register_default_hanlder: Determines whether to register default signal handler for SIGINT signal.
     *                                       Defaults to false.
     * @note: The default handler for SIGINT signal is @ref ral::lib::services::SignalHandler::default_signal_handler.
     */
    SignalHandler(bool register_default_hanlder = false);
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
     * This method uses @ref ral::lib::services::SignalHandler::default_signal_handler
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
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_SERVICES_ERROR_HANDLING_SIGNAL_HANDLER_H_
#endif /* RMAX_APPS_LIB_LIB_SERVICES_ERROR_HANDLING_SIGNAL_HANDLER_H_ */
