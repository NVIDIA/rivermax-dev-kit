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

#ifndef RDK_APPS_RMAX_IPO_RECEIVER_RMAX_IPO_RECEIVER_H_
#define RDK_APPS_RMAX_IPO_RECEIVER_RMAX_IPO_RECEIVER_H_

#include "rdk/apps/rmax_receiver_base.h"
#include "rdk/io_node/receivers/ipo_receiver_io_node.h"

using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
{
namespace apps
{
namespace rmax_ipo_receiver
{

/**
 * @brief: Configuration settings for Rivermax IPO Receiver.
 */
struct IPOReceiverSettings : AppSettings
{
public:
    static constexpr uint32_t DEFAULT_NUM_OF_PACKETS_IN_CHUNK = 262144;
    void init_default_values() override;

    uint32_t max_path_differential_us;
    bool is_extended_sequence_number;
    size_t max_chunk_size;
};

/**
 * @brief: Validator for Rivermax IPO Receiver settings.
 */
class IPOReceiverSettingsValidator : public ISettingsValidator<IPOReceiverSettings>
{
public:
     ReturnStatus validate(const std::shared_ptr<IPOReceiverSettings>& settings) const override;
};

/**
 * @brief: CLI settings Builder for Rivermax IPO Receiver.
 */
class IPOReceiverCLISettingsBuilder : public CLISettingsBuilder<IPOReceiverSettings>
{
public:
    static constexpr int USECS_IN_SECOND = 1000000;
    /**
     * @brief: IPOReceiverCLISettingsBuilder constructor.
     *
     * @param [in] argc: Number of CLI arguments.
     * @param [in] argv: CLI arguments strings array.
     * @param [in] app_description: Application description string for the CLI usage.
     * @param [in] app_examples: Application examples string for the CLI usage.
     */
    IPOReceiverCLISettingsBuilder(int argc, const char** argv,
        const std::string& app_description,
        const std::string& app_examples,
        std::shared_ptr<ISettingsValidator<IPOReceiverSettings>> validator) :
        CLISettingsBuilder<IPOReceiverSettings>(argc, argv, app_description, app_examples, std::move(validator)) {}
    virtual ~IPOReceiverCLISettingsBuilder() = default;
protected:
    ReturnStatus add_cli_options(std::shared_ptr<IPOReceiverSettings>& settings) override;
};

using IPOReceiverExternalSettingsBuilder = ExternalSettingsBuilder<IPOReceiverSettings>;
/**
 *
 * @brief: IPO Receiver application.
 *
 * This is an example of usage application for Rivermax Inline Packet Ordering RX API.
 */
class IPOReceiverApp : public RmaxReceiverBaseApp
{
private:
    /* Settings builder pointer */
    std::shared_ptr<ISettingsBuilder<IPOReceiverSettings>> m_settings_builder;
    /* Application settings pointer */
    std::shared_ptr<IPOReceiverSettings> m_ipo_receiver_settings;
    /* Network recv flows */
    std::vector<std::vector<ReceiveFlow>> m_flows;
public:
    /**
     * @brief: IPOReceiverApp class constructor.
     *
     * @param [in] settings_builder: Settings builder pointer.
     */
    IPOReceiverApp(std::shared_ptr<ISettingsBuilder<IPOReceiverSettings>> settings_builder);
    /**
     * @brief: IPOReceiverApp class destructor.
     */
    virtual ~IPOReceiverApp() = default;
    /**
     * @brief: Get IPO streams total statistics.
     *
     * This function hides the base class function and provides default template arguments.
     *
     * @return: A vector of @ref IPORXStatistics.
     */
    std::vector<IPORXStatistics> get_streams_total_statistics() const {
        return RmaxReceiverBaseApp::get_streams_total_statistics<IPORXStatistics, AppIPOReceiveStream>();
    }

private:
    ReturnStatus initialize_app_settings() final;
    ReturnStatus initialize_connection_parameters() final;
    void configure_network_flows() final;
    void initialize_receive_io_nodes() final;
    void run_receiver_threads() final;
};

} // namespace rmax_ipo_receiver
} // namespace apps
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_APPS_RMAX_IPO_RECEIVER_RMAX_IPO_RECEIVER_H_ */
