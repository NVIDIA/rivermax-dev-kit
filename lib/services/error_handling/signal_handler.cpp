/*
 * Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#include <atomic>
#include <iostream>
#include <climits>
#include <csignal>

#include "services/error_handling/signal_handler.h"

using namespace ral::lib::services;

std::atomic_int SignalHandler::s_received_signal(INT_MIN);
signal_handlers_map_t SignalHandler::s_signal_handlers_map;

SignalHandler::SignalHandler(bool register_default_hanlder)
{
    if (register_default_hanlder) {
        initialize_default_signal_handlers();
    }
}

void SignalHandler::initialize_default_signal_handlers()
{
    register_signal_handler_callback(SIGINT, SignalHandler::default_signal_handler);
}

void SignalHandler::default_signal_handler(int signal)
{
    std::cout << "\n\n<--- Signal (" << signal << ") received --->\n\n\n";
}

void SignalHandler::signal_handler_arbiter(int signal)
{
    auto iter = SignalHandler::s_signal_handlers_map.find(signal);
    if (iter != SignalHandler::s_signal_handlers_map.end()) {
        iter->second(signal);
    } else {
        SignalHandler::default_signal_handler(signal);
    }
    SignalHandler::s_received_signal.store(signal);
}

void SignalHandler::register_signal_handler_callback(int signal, signal_handler_t handler)
{
    SignalHandler::s_signal_handlers_map[signal] = handler;
    std::signal(signal, SignalHandler::signal_handler_arbiter);
}
