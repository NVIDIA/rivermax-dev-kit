/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef RDK_EXAMPLES_BASE_EXAMPLE_H_
#define RDK_EXAMPLES_BASE_EXAMPLE_H_

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

#include "rdk/facade.h"

using namespace rivermax::dev_kit::services;

namespace rivermax
{
namespace dev_kit
{
namespace examples
{
/**
 * @brief: Base class for Rivermax Dev Kit examples.
 *
 * This class serves as a foundation for all Rivermax Dev Kit examples, providing
 * common functionality such as command line argument parsing, error handling, and
 * a standardized execution flow. Derived classes should implement the @ref operator()
 * method that contains the main example logic.
 */
class BaseExample
{
public:
    /**
     * @brief: Main function to run the example with error handling.
     *
     * This method provides a common entry point for all examples with standardized
     * error handling. It calls the derived class's @ref operator() method within a
     * try-catch block to ensure proper error handling.
     *
     * @param [in] argc: Number of command line arguments.
     * @param [in] argv: Array of command line argument strings.
     *
     * @return: @ref EXIT_SUCCESS on success, @ref EXIT_FAILURE on failure.
     */
    virtual int run(int argc, const char* argv[]);

protected:
    /** Rivermax Dev Kit facade for accessing services */
    RivermaxDevKitFacade& m_facade;
    /** Settings object for storing parsed CLI values */
    std::shared_ptr<AppSettings> m_settings;
    /** CLI parser manager for handling command line options */
    std::shared_ptr<CLIParserManager> m_cli_parser_manager;
    /** Brief description of the example application */
    std::string m_description;
    /** Usage examples for the command line */
    std::string m_usage;

    /**
     * @brief: Constructor for the base example class.
     *
     * @param [in] description: Brief description of the example.
     * @param [in] usage: Usage examples for the command line binary.
     */
    BaseExample(const std::string& description, const std::string& usage);
    /**
     * @brief: Destructor for the base example class.
     */
    virtual ~BaseExample() = default;
    /**
     * @brief: Entry point for the example execution.
     *
     * This function should be implemented by derived classes to contain the main
     * example code logic. It will be called by the @ref run method within a try-catch block.
     *
     * @return: Status of the example execution.
     */
    virtual ReturnStatus operator()() = 0;
    /**
     * @brief: Initializes example common default settings.
     *
     * Override this method to set application-specific default settings.
     * It will be called as part of the initialization process before CLI parsing.
     */
    virtual void initialize_common_default_settings();
    /**
     * @brief: Adds CLI options and/or arguments to the parser.
     *
     * Use this method to add CLI options to the example binary by using @ref m_cli_parser_manager->add_option.
     * The base implementation adds the local IP argument. Override to add additional options.
     * It will be called as part of the initialization process.
     */
    virtual void add_cli_options();
    /**
     * @brief: Does post CLI parsing initialization.
     *
     * Use this method to do any needed post CLI parsing example initialization,
     * such as extracting values from the parsed settings.
     * It will be called as part of the initialization process.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus post_cli_parse_initialization() { return ReturnStatus::success; }

private:
    /**
     * @brief: Initializes the example with CLI parsing and other setup.
     *
     * This method runs example initialization flow using the other methods in this
     * interface. It initializes the CLI parser, parses command line arguments,
     * and performs post-parsing initialization.
     *
     * @param [in] argc: Number of command line arguments.
     * @param [in] argv: Array of command line argument strings.
     *
     * @return: Status of the operation.
     */
    ReturnStatus initialize(int argc, const char* argv[]);
};

/**
 * @brief: Common templated main function for examples.
 *
 * This function wraps the example execution with proper exception handling,
 * ensuring that exceptions thrown during example construction are caught.
 *
 * @tparam ExampleType: The type of the example class, must inherit from BaseExample.
 * @param [in] argc: Number of command line arguments.
 * @param [in] argv: Array of command line arguments.
 *
 * @return: Exit status of the example.
 */
template <typename ExampleType>
int common_example_main(int argc, const char* argv[])
{
    static_assert(std::is_base_of<BaseExample, ExampleType>::value,
        "ExampleType must inherit from BaseExample");

    try {
        return ExampleType().run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        return EXIT_FAILURE;
    }
}

} // namespace examples
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_EXAMPLES_BASE_EXAMPLE_H_ */
