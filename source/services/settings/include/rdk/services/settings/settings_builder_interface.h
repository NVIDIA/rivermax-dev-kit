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

#ifndef RDK_SERVICES_SETTINGS_SETTINGS_BUILDER_INTERFACE_H_
#define RDK_SERVICES_SETTINGS_SETTINGS_BUILDER_INTERFACE_H_

#include <string>
#include <memory>

#include "rdk/services/error_handling/return_status.h"
#include "rdk/services/cli/cli_manager.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Instructions for creating a settings builder for a new application.
 *
 * To create a settings builder for a new application using the provided framework, follow these steps:
 *
 * 1. Define the Configuration Structure: (Optional)
 *    - Create a structure that inherits from `AppSettings` to hold your application's specific configuration settings.
 *    - Override the void init_default_values() function to set default values for your settings.
 *
 * 2. Create Validator Class:
 *    - Define a class that validates the settings for your application.
 *      This class should have a method to validate the settings:
 *          ReturnStatus validate(const std::shared_ptr<SettingsType>& settings) const
 *
 * 4. Implement Settings Builder Classes:
 *    - Implement desired settings builder classes by extending `CLISettingsBuilder`,
 *      and `ExternalSettingsBuilder` with your custom initializer and validator.
 */

/**
 * @brief: Interface for validating application settings.
 *
 * This interface provides a method for validating application settings.
 *
 * @tparam SettingsType: The type of the settings to validate.
 */
template <typename SettingsType>
class ISettingsValidator
{
public:
    /**
     * @brief: Destructor for ISettingsValidator.
     */
    virtual ~ISettingsValidator() = default;
    /**
     * @brief: Validates the application settings.
     *
     * @param [in] settings: A shared pointer to the application settings.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus validate(const std::shared_ptr<SettingsType>& settings) const = 0;
};

/**
 * @brief: Interface for building application settings.
 *
 * This interface provides a method for building application settings.
 *
 * @tparam SettingsType: The type of the settings to build.
 */
template <typename SettingsType>
class ISettingsBuilder
{
public:
    /**
     * @brief: Destructor for ISettingsBuilder.
     */
    virtual ~ISettingsBuilder() = default;
    /**
     * @brief: Builds the application settings.
     *
     * @param [in] settings: A shared pointer to the application settings.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus build(std::shared_ptr<SettingsType> settings) = 0;
};

/**
 * @brief: Base class for settings builders.
 *
 * This class provides common functionality for settings builders.
 *
 * @tparam SettingsType: The type of the settings to build.
 */
template <typename SettingsType>
class SettingsBuilderBase : public ISettingsBuilder<SettingsType>
{
protected:
    std::shared_ptr<ISettingsValidator<SettingsType>> m_validator;
public:
    /**
     * @brief: Constructor for SettingsBuilderBase.
     *
     * @param [in] validator: A shared pointer to the settings validator.
     */
    SettingsBuilderBase(std::shared_ptr<ISettingsValidator<SettingsType>> validator) :
        m_validator(std::move(validator)) {}
};

/**
 * @brief: Command line settings builder.
 *
 * This class provides functionality for building settings from command line arguments.
 *
 * @tparam SettingsType: The type of the settings to build.
 */
template <typename SettingsType>
class CLISettingsBuilder : public SettingsBuilderBase<SettingsType>
{
protected:
    std::string m_app_description;
    std::string m_app_examples;
    int m_argc;
    const char** m_argv;
    std::shared_ptr<CLIParserManager> m_cli_parser_manager = nullptr;
public:
    /**
     * @brief: Constructor for CLISettingsBuilder.
     *
     * @param [in] argc: The number of command line arguments.
     * @param [in] argv: The command line arguments.
     * @param [in] app_description: The description of the application.
     * @param [in] app_examples: Examples of how to use the application.
     * @param [in] validator: A shared pointer to the settings validator.
     */
    CLISettingsBuilder(int argc, const char** argv,
        const std::string& app_description,
        const std::string& app_examples,
        std::shared_ptr<ISettingsValidator<SettingsType>> validator) :
        SettingsBuilderBase<SettingsType>(std::move(validator)),
        m_app_description(app_description), m_app_examples(app_examples),
        m_argc(argc), m_argv(argv) {}
    /**
     * @brief: Destructor for CLISettingsBuilder.
     */
    virtual ~CLISettingsBuilder() = default;
    /**
     * @brief: Builds the application settings.
     *
     * @param [in] settings: A shared pointer to the application settings.
     *
     * @return: Status of the operation.
     */
    ReturnStatus build(std::shared_ptr<SettingsType> settings) override;
protected:
    /**
     * @brief: Adds CLI options and/or arguments to the parser.
     *
     * Override this method to add CLI options to the application, by using @ref m_cli_parser_manager->add_option.
     * It will be called as part of the @ref CLISettingsBuilder::build process.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus add_cli_options(std::shared_ptr<SettingsType>& settings) { return ReturnStatus::success; }
    /**
     * @brief: Parses the CLI arguments.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus parse_cli() {
        return m_cli_parser_manager == nullptr ? ReturnStatus::failure :
            m_cli_parser_manager->parse_cli(m_argc, m_argv);
    }
    /**
     * @brief: Initializes the CLI parser manager.
     *
     * @param [in] settings: A shared pointer to the application settings.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus init_cli_parser_manager(std::shared_ptr<SettingsType> settings);
};

/**
 * @brief: External settings builder.
 *
 * This class provides functionality for building settings from external sources.
 *
 * @tparam SettingsType: The type of the settings to build.
 */
template <typename SettingsType>
class ExternalSettingsBuilder : public SettingsBuilderBase<SettingsType>
{
private:
    std::shared_ptr<SettingsType> m_external_settings;
public:
    /**
     * @brief: Constructor for ExternalSettingsBuilder.
     *
     * @param [in] external_settings: A shared pointer to the external settings.
     * @param [in] validator: A shared pointer to the settings validator.
     */
    ExternalSettingsBuilder(std::shared_ptr<SettingsType> external_settings,
        std::shared_ptr<ISettingsValidator<SettingsType>> validator) :
        SettingsBuilderBase<SettingsType>(validator), m_external_settings(external_settings) {}
    /**
     * @brief: Destructor for ExternalSettingsBuilder.
     */
    ~ExternalSettingsBuilder() = default;
    /**
     * @brief: Builds the application settings.
     *
     * @param [in] settings: A shared pointer to the application settings.
     *
     * @return: Status of the operation.
     */
    ReturnStatus build(std::shared_ptr<SettingsType> settings) override {
        *settings = *m_external_settings;
        if (this->m_validator) {
            return this->m_validator->validate(settings);
        }
        return ReturnStatus::success;
    }
};

/**
 * @brief: Conversion settings builder.
 *
 * This class provides functionality for converting settings from one type to another.
 *
 * @tparam SourceSettingsType: The type of the source settings to convert from.
 * @tparam TargetSettingsType: The type of the settings to build.
 */
template <typename SourceSettingsType, typename TargetSettingsType>
class ConversionSettingsBuilder : public SettingsBuilderBase<TargetSettingsType>
{
private:
    std::shared_ptr<SourceSettingsType> m_source_settings;
public:
    /**
     * @brief: Constructor for @ref ConversionSettingsBuilder.
     *
     * @param [in] source_settings: A shared pointer to the source settings.
     * @param [in] validator: A shared pointer to the target settings validator.
     */
    ConversionSettingsBuilder(std::shared_ptr<SourceSettingsType> source_settings,
        std::shared_ptr<ISettingsValidator<TargetSettingsType>> validator) :
        SettingsBuilderBase<TargetSettingsType>(validator), m_source_settings(source_settings) {}
    /**
     * @brief: Destructor for @ref ConversionSettingsBuilder.
     */
    ~ConversionSettingsBuilder() = default;
    /**
     * @brief: Builds the application settings by converting from source settings.
     *
     * @param [in] target_settings: A shared pointer to the target application settings.
     *
     * @return: Status of the operation.
     */
    ReturnStatus build(std::shared_ptr<TargetSettingsType> target_settings) override;
protected:
    /**
     * @brief: Converts source settings to target settings.
     *
     * Override this method to implement the conversion logic from source settings to target settings.
     *
     * @param [in] source_settings: A shared pointer to the source settings.
     * @param [out] target_settings: A shared pointer to the target settings.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus convert_settings(const std::shared_ptr<SourceSettingsType>& source_settings,
                                          std::shared_ptr<TargetSettingsType>& target_settings) = 0;
};

template <typename SourceSettingsType, typename TargetSettingsType>
ReturnStatus ConversionSettingsBuilder<SourceSettingsType, TargetSettingsType>::build(
    std::shared_ptr<TargetSettingsType> target_settings) {
    if (m_source_settings == nullptr) {
        std::cerr << "Source settings are null" << std::endl;
        return ReturnStatus::failure;
    }
    target_settings->init_default_values();
    auto rc = convert_settings(m_source_settings, target_settings);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to convert settings" << std::endl;
        return rc;
    }

    if (this->m_validator) {
        return this->m_validator->validate(target_settings);
    }
    return ReturnStatus::success;
}

template <typename SettingsType>
ReturnStatus CLISettingsBuilder<SettingsType>::build(std::shared_ptr<SettingsType> settings)
{
    settings->init_default_values();

    auto rc = this->init_cli_parser_manager(settings);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to initialize CLI parser manager" << std::endl;
        return rc;
    }

    rc = this->add_cli_options(settings);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to add CLI options" << std::endl;
        return rc;
    }

    rc = this->parse_cli();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to parse CLI" << std::endl;
        return rc;
    } else if (rc == ReturnStatus::success_cli_help) {
        return rc;
    }

    if (this->m_validator) {
        rc = this->m_validator->validate(settings);
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to validate settings" << std::endl;
            return rc;
        }
    }

    return ReturnStatus::success;
}

template <typename SettingsType>
ReturnStatus CLISettingsBuilder<SettingsType>::init_cli_parser_manager(std::shared_ptr<SettingsType> settings)
{
    m_cli_parser_manager = std::make_shared<CLIParserManager>(m_app_description, m_app_examples, settings);

    auto rc = m_cli_parser_manager->initialize();
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to initialize CLI manager" << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_SETTINGS_SETTINGS_BUILDER_INTERFACE_H_ */
