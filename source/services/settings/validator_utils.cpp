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

#include <string>
#include <cstring>
#include <thread>

#include "rt_threads.h"

#include "rdk/services/settings/validator_utils.h"

ReturnStatus ValidatorUtils::validate_ip4_address(const std::string& ip)
{
    std::string validation_error = CLI::ValidIPV4(ip);
    if (!validation_error.empty()) {
        std::cerr << "IP " << ip << " is not a valid IPv4 address: " << validation_error << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

ReturnStatus ValidatorUtils::validate_ip4_address(const std::vector<std::string>& ips)
{
    for (const auto& ip : ips) {
        ReturnStatus status = validate_ip4_address(ip);
        if (status != ReturnStatus::success) {
            return status;
        }
    }
    return ReturnStatus::success;
}

ReturnStatus ValidatorUtils::validate_ip4_port(uint16_t port)
{
    CLI::Range validator(MIN_PORT, MAX_PORT);
    std::string validation_error = validator(std::to_string(port));
    if (!validation_error.empty()) {
        std::cerr << "IP4 Port error: " << validation_error << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

ReturnStatus ValidatorUtils::validate_ip4_port(const std::vector<uint16_t>& ports)
{
    for (const auto& port : ports) {
        ReturnStatus status = validate_ip4_port(port);
        if (status != ReturnStatus::success) {
            return status;
        }
    }
    return ReturnStatus::success;
}

ReturnStatus ValidatorUtils::validate_core(int core)
{
    if (core == CPU_NONE) {
        return ReturnStatus::success;
    }
    unsigned int max_cores = std::thread::hardware_concurrency();
    if (core < 0 || core >= static_cast<int>(max_cores)) {
        std::cerr << "Core " << core << " is out of valid range [0, " << max_cores - 1 << "]" << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

ReturnStatus ValidatorUtils::validate_core(const std::vector<int>& cores)
{
    for (const auto& core : cores) {
        ReturnStatus status = validate_core(core);
        if (status != ReturnStatus::success) {
            return status;
        }
    }
    return ReturnStatus::success;
}
