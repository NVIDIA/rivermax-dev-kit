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

#include <iostream>

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

    return ReturnStatus::success;
}