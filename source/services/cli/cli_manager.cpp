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

#include <memory>
#include <string>
#include <functional>

#include "CLI/CLI.hpp"
#include "rt_threads.h"

#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/cli/cli_manager.h"
#include "rdk/services/cli/options.h"
#include "rdk/services/cli/validators.h"

using namespace rivermax::dev_kit::services;

CLIParserManager::CLIParserManager(
    const std::string& app_description,
    const std::string& app_examples,
    std::shared_ptr<AppSettings> app_settings) :
    m_app_settings(std::move(app_settings)),
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
