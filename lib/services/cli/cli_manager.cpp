/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

#include "CLI/CLI.hpp"
#include "rt_threads.h"

#include "services/cli/cli_manager.h"
#include "services/cli/validators.h"
#include "services/cli/options.h"
#include "services/error_handling/return_status.h"

using namespace ral::lib::services;

CLIParserManager::CLIParserManager(
    const std::string& app_description,
    const std::string& app_examples,
    std::shared_ptr<AppSettings> app_settings) :
    m_app_settings(app_settings),
    m_app_description(app_description),
    m_app_examples(app_examples)
{

}

ReturnStatus CLIParserManager::initialize()
{
    try {
        m_parser = std::make_shared<CLI::App>(m_app_description);
    }
    catch (const CLI::BadNameString& error) {
        std::cerr << "Failed to initialize CLI parser; error:" << error.get_exit_code() << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus CLIParserManager::parse_cli(int argc, const char* argv[])
{
    try {
        m_parser->footer(m_app_examples);
        m_parser->parse(argc, argv);
    }
    catch (const CLI::CallForHelp & error) {
        m_parser->exit(error);
        return ReturnStatus::success_cli_help;
    }
    catch (const CLI::ParseError & error) {
        m_parser->exit(error);
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

CLI::Option* CLIParserManager::add_option(const std::string& option_name)
{
    auto iter = CLIParserManager::s_cli_opt_fuctory.find(option_name);
    if (iter != CLIParserManager::s_cli_opt_fuctory.end()) {
        try {
            return iter->second(m_parser, m_app_settings);
        }
        catch (const CLI::BadNameString& error) {
            std::cerr << "Failed to add CLI option [" << option_name << "]; error: " << error.get_exit_code() << std::endl;
            return nullptr;
        }
    } else {
        return nullptr;
    }
}
