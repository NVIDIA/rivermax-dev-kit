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

#ifndef RDK_SERVICES_SETTINGS_VALIDATOR_UTILS_H_
#define RDK_SERVICES_SETTINGS_VALIDATOR_UTILS_H_

#include <string>
#include <vector>

#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/cli/cli_manager.h"

using namespace rivermax::dev_kit::services;

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Utility class for validating application settings.
 *
 * This class provides static methods for validating various application settings,
 * such as IP addresses and CPU cores.
 */
class ValidatorUtils {
public:
    /**
     * @brief: Validate a list of IPv4 addresses.
     *
     * @param [in] ips: A vector of strings representing the IPv4 addresses to validate.
     *
     * @return: Returns status of the operation.
     */
    static ReturnStatus validate_ip4_address(const std::vector<std::string>& ips);
    /**
     * @brief: Validate a single IPv4 address.
     *
     * @param [in] ip: A string representing the IPv4 address to validate.
     *
     * @return: Returns status of the operation.
     */
    static ReturnStatus validate_ip4_address(const std::string& ip);
    /**
     * @brief: Validate a single IP4 port number.
     *
     * @param [in] port: An integer representing the IP4 port number to validate.
     *
     * @return: Returns status of the operation.
     */
    static ReturnStatus validate_ip4_port(uint16_t port);
    /**
     * @brief: Validate a list of IP4 port numbers.
     *
     * @param [in] ports: A vector of integers representing the IP4 port numbers to validate.
     *
     * @return: Returns status of the operation.
     */
    static ReturnStatus validate_ip4_port(const std::vector<uint16_t>& ports);
    /**
     * @brief: Validate a CPU core number.
     *
     * @param [in] core: An integer representing the CPU core number to validate.
     *
     * @return: Returns status of the operation.
     */
    static ReturnStatus validate_core(int core);
    /**
     * @brief: Validate a list of CPU core numbers.
     *
     * @param [in] cores: A vector of integers representing the CPU core numbers to validate.
     *
     * @return: Returns status of the operation.
     */
    static ReturnStatus validate_core(const std::vector<int>& cores);
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_SETTINGS_VALIDATOR_UTILS_H_ */
