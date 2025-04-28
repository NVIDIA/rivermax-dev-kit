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

#include <cstring>
#include <iostream>

#include "rdk/facade.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/examples/base_example.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::examples;

BaseExample::BaseExample(const std::string& description, const std::string& usage) :
    m_facade(rivermax::dev_kit::RivermaxDevKitFacade::get_instance()),
    m_settings(std::make_shared<AppSettings>()),
    m_description(description),
    m_usage(usage)
{
}

int BaseExample::run(int argc, const char* argv[])
{
    try {
        auto status = initialize(argc, argv);
        if (status == ReturnStatus::success_cli_help) {
            return EXIT_SUCCESS;
        } else if (status != ReturnStatus::success) {
            std::cerr << "Failed to initialize example" << std::endl;
            return EXIT_FAILURE;
        }

        status = this->operator()();
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to run example with status: " << static_cast<int>(status) << std::endl;
            return EXIT_FAILURE;
        }
    } catch (const std::exception& error) {
        std::cerr << "Failed to run example with exception: " << error.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Failed to run example with unknown exception" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

void BaseExample::initialize_common_default_settings()
{
    m_settings->init_default_values();
}

void BaseExample::add_cli_options()
{
    m_cli_parser_manager->add_option(CLIOptStr::LOCAL_IP);
}

ReturnStatus BaseExample::initialize(int argc, const char* argv[])
{
    m_cli_parser_manager = m_facade.get_cli_parser_manager(m_description, m_usage, m_settings);

    auto status = m_cli_parser_manager->initialize();
    if (status != ReturnStatus::success) {
        std::cerr << "Failed to initialize CLI manager" << std::endl;
        return ReturnStatus::failure;
    }

    initialize_common_default_settings();
    add_cli_options();

    status = m_cli_parser_manager->parse_cli(argc, argv);
    if (status == ReturnStatus::success_cli_help) {
        return ReturnStatus::success_cli_help;
    } else if (status == ReturnStatus::failure) {
        std::cerr << "Failed to parse CLI options" << std::endl;
        return ReturnStatus::failure;
    }

    status = post_cli_parse_initialization();
    if (status != ReturnStatus::success) {
        std::cerr << "Failed to do post CLI parse initialization" << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}
