/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <iostream>
#include <chrono>
#include <thread>

#include "rivermax_api.h"
#include "clock.h"

using namespace ral::lib::services;

ReturnStatus ral::lib::services::set_rivermax_user_clock(rmx_user_clock_handler handler, void* ctx)
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

ReturnStatus ral::lib::services::set_rivermax_ptp_clock(const rmx_device_iface* device_iface)
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