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

#include <atomic>
#include <iostream>
#include <climits>
#include <csignal>

#include "rdk/services/error_handling/signal_handler.h"

using namespace rivermax::dev_kit::services;

std::atomic_int SignalHandler::s_received_signal(INT_MIN);
signal_handlers_map_t SignalHandler::s_signal_handlers_map;

SignalHandler::SignalHandler(bool register_default_handler)
{
    if (register_default_handler) {
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
    SignalHandler::s_signal_handlers_map[signal] = std::move(handler);
    std::signal(signal, SignalHandler::signal_handler_arbiter);
}
