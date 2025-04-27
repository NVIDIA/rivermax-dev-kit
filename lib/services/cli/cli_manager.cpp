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
    m_parser(new CLI::App(app_description)),
    m_app_examples(app_examples)
{
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
        return iter->second(m_parser, m_app_settings);
    }
    else {
        return nullptr;
    }
}
