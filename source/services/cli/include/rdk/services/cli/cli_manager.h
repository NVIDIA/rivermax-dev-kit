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

#ifndef RDK_SERVICES_CLI_CLI_MANAGER_H_
#define RDK_SERVICES_CLI_CLI_MANAGER_H_

#include <string>
#include <memory>

#include "CLI/CLI.hpp"

#include "rdk/services/cli/options.h"
#include "rdk/services/error_handling/return_status.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 * @brief: Command line parser manager.
 *
 * The class will manage all command line related operation.
 */
class CLIParserManager
{
public:
    static cli_opt_factory_map_t s_cli_opt_fuctory;
private:
    std::shared_ptr<AppSettings> m_app_settings;
    CLI::App_p m_parser;
    std::string m_app_description;
    std::string m_app_examples;
public:
    /**
     * @brief: CLIParserManager constructor.
     *
     * @param [in] app_description: Application description string for the CLI usage.
     * @param [in] app_examples: Application examples string for the CLI usage.
     * @param [in] app_settings: Application settings.
     */
    CLIParserManager(const std::string& app_description, const std::string& app_examples,
                     std::shared_ptr<AppSettings> app_settings);

    /**
     * @brief: Initializes the CLI manager.
     *
     * During initialization, required ressources will be initialized
     * and these may fail. Returned value will indicate how initialization went.
     *
     * @return: Status of operation
     */
    ReturnStatus initialize();

    /**
     * @brief: Returns CLI parser object.
     *
     * The user of this method can use the parser object to add custom options
     * to the parser following CLI11 library usage.
     *
     * @return: Shared pointer to the application parser object.
     */
    CLI::App_p get_parser() { return m_parser; }
    /**
     * @brief: Parses command line options.
     *
     * The method will handle parsing command line options.
     *
     * @param [in] argc: Number of CLI arguments.
     * @param [in] argv: CLI arguments strings array.
     *
     * @return: Status of the operation.
     */
    ReturnStatus parse_cli(int argc, const char* argv[]);
    /**
     * @brief: Add CLI option/argument to the parser.
     *
     * The method will handle adding CLI options or arguments from a dedicated
     * predefined options. The available options can be seen in
     * @ref CLIParserManager::s_cli_opt_fuctory in the options.cpp file.
     *
     * @note: The user can use the returned CLI::Option* pointer to alter it.
     *
     * @param [in] option_name: The name of the option,
     *                          available options are in @ref CLIOptStr.
     *
     * @return: Pointer to the CLI option.
     */
    CLI::Option* add_option(const std::string& option_name);
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_CLI_CLI_MANAGER_H_ */
