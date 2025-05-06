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

#include <iomanip>
#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>

#include <rivermax_api.h>

#include "rdk/services/utils/clock.h"

using namespace rivermax::dev_kit::services;

ReturnStatus rivermax::dev_kit::services::set_rivermax_user_clock(rmx_user_clock_handler handler, void* ctx)
{
    rmx_user_clock_params clock_params;

    rmx_init_user_clock(&clock_params);
    rmx_set_user_clock_handler(&clock_params, handler);
    rmx_set_user_clock_context(&clock_params, ctx);
    rmx_status status = rmx_use_user_clock(&clock_params);

    if (status != RMX_OK) {
        std::cerr << "Failed to set Rivermax user clock with status:" << status << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus rivermax::dev_kit::services::set_rivermax_ptp_clock(const rmx_device_iface* device_iface)
{
    rmx_ptp_clock_params clock_params;

    rmx_init_ptp_clock(&clock_params);
    rmx_set_ptp_clock_device(&clock_params, device_iface);
    rmx_status status = rmx_use_ptp_clock(&clock_params);

    if (status != RMX_OK) {
        std::cerr << "Failed to set Rivermax PTP clock with status:" << status << std::endl;
        return ReturnStatus::failure;
    }

    while ((status = rmx_check_clock_steady()) == RMX_BUSY) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return (status == RMX_OK) ? ReturnStatus::success : ReturnStatus::failure;
}

ReturnStatus rivermax::dev_kit::services::get_rivermax_ptp_time_ns(uint64_t& ptp_time_ns)
{
    rmx_status status = rmx_get_time(RMX_TIME_PTP, &ptp_time_ns);
    return (status == RMX_OK) ? ReturnStatus::success : ReturnStatus::failure;
}

std::string rivermax::dev_kit::services::ptp_to_string(uint64_t timestamp)
{
    constexpr size_t ns_in_sec = std::chrono::nanoseconds{ std::chrono::seconds{ 1 } }.count();
    uint32_t nsec = timestamp % ns_in_sec;
    std::chrono::nanoseconds ns(timestamp);
    std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> tp(ns);
    auto tp_system = std::chrono::time_point_cast<std::chrono::system_clock::duration>(tp);
    auto time = std::chrono::system_clock::to_time_t(tp_system);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setw (9) << std::setfill ('0') << nsec;
    return oss.str();
}
