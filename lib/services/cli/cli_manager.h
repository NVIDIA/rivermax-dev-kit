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

#ifndef RMAX_APPS_LIB_LIB_SERVICES_CLI_CLI_MANAGER_H_

#include <string>
#include <cstddef>
#include <memory>

#include "CLI/CLI.hpp"

#include "services/cli/options.h"
#include "services/utils/defs.h"
#include "services/error_handling/return_status.h"

namespace ral
{
namespace lib
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
    std::shared_ptr<CLI::App> m_parser;
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
     * @brief: Returns CLI parser object.
     *
     * The user of this method can use the parser object to add custom options
     * to the parser following CLI11 library usage.
     *
     * @return: Shared pointer to the application parser object.
     */
    std::shared_ptr<CLI::App> get_parser() { return m_parser; }
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
     * @ref ral::lib::services::CLIParserManager::s_cli_opt_fuctory in the options.cpp file.
     *
     * @note: The user can use the returned CLI::Option* pointer to alter it.
     *
     * @param [in] option_name: The name of the option,
     *                          available options are in @ref ral::lib::services::CLIOptStr.
     *
     * @return: Pointer to the CLI option.
     */
    CLI::Option* add_option(const std::string& option_name);
};

} // namespace services
} // namespace lib
} // namespace ral


#define RMAX_APPS_LIB_LIB_SERVICES_CLI_CLI_MANAGER_H_
#endif /* RMAX_APPS_LIB_LIB_SERVICES_CLI_CLI_MANAGER_H_ */
