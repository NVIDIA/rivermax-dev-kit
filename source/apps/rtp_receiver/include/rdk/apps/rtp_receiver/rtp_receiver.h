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

#ifndef RDK_APPS_RTP_RECEIVER_RTP_RECEIVER_H_
#define RDK_APPS_RTP_RECEIVER_RTP_RECEIVER_H_

#include "rdk/apps/receiver_base_app.h"

namespace rivermax
{
namespace dev_kit
{
namespace apps
{
namespace rtp_receiver
{

/**
 * @brief: Configuration settings for Rivermax Dev Kit RTP Receiver.
 */
struct RTPReceiverSettings : AppSettings
{
public:
    static constexpr uint32_t DEFAULT_NUM_OF_PACKETS_IN_CHUNK = 262144;
    void init_default_values() override;

    bool is_extended_sequence_number;
};

/**
 * @brief: Validator for Rivermax Dev Kit RTP Receiver settings.
 */
class RTPReceiverSettingsValidator : public ISettingsValidator<RTPReceiverSettings>
{
public:
     ReturnStatus validate(const RTPReceiverSettings& settings) const override;
};

/**
 * @brief: CLI settings Builder for Rivermax Dev Kit RTP Receiver.
 */
class RTPReceiverCLISettingsBuilder : public CLISettingsBuilder<RTPReceiverSettings>
{
public:
    /**
     * @brief: RTPReceiverCLISettingsBuilder constructor.
     *
     * @param [in] argc: Number of CLI arguments.
     * @param [in] argv: CLI arguments strings array.
     * @param [in] app_description: Application description string for the CLI usage.
     * @param [in] app_examples: Application examples string for the CLI usage.
     */
    RTPReceiverCLISettingsBuilder(int argc, const char** argv,
        const std::string& app_description,
        const std::string& app_examples,
        const ISettingsValidator<RTPReceiverSettings>& validator) :
        CLISettingsBuilder<RTPReceiverSettings>(argc, argv, app_description, app_examples, validator) {}
    virtual ~RTPReceiverCLISettingsBuilder() = default;
protected:
    ReturnStatus add_cli_options(RTPReceiverSettings& settings) override;
};

using RTPReceiverUserProvidedSettingsBuilder = UserProvidedSettingsBuilder<RTPReceiverSettings>;
/**
 * @brief: RTP receiver application.
 *
 * This is an example of application that uses Rivermax Dev Kit to receive multiple
 * RTP streams. It's intended to be as simple as possible so it doesn't include
 * redundancy or reordering logic. The number of dropped packets are calculated
 * using RTP sequence numbers. Depending on a command line parameters, the
 * application supports standard RTP sequence numbers or extended sequence
 * numbers as defined in SMPTE ST 2110-20.
 */
class RTPReceiverApp : public ReceiverBaseApp
{
private:
    /* Settings builder pointer */
    std::unique_ptr<ISettingsBuilder<RTPReceiverSettings>> m_settings_builder;
    /* Application settings pointer */
    std::shared_ptr<RTPReceiverSettings> m_rtp_receiver_settings;
    /* Network recv flows */
    std::vector<ReceiveFlow> m_flows;
public:
    /**
     * @brief: RTPReceiverApp class constructor.
     *
     * @param [in] settings_builder: Settings builder pointer.
     */
    RTPReceiverApp(std::unique_ptr<ISettingsBuilder<RTPReceiverSettings>> settings_builder);
    /**
     * @brief: RTPReceiverApp class destructor.
     */
    virtual ~RTPReceiverApp() = default;
    /**
     * @brief: Get RTP streams total statistics.
     *
     * This function hides the base class function and provides default template arguments.
     *
     * @return: A vector of @ref RXStatistics.
     */
    std::vector<RXStatistics> get_streams_total_statistics() const {
        return ReceiverBaseApp::get_streams_total_statistics<RXStatistics, AppRTPReceiveStream>();
    }

private:
    ReturnStatus initialize_app_settings() final;
    void configure_network_flows() final;
    void initialize_receive_io_nodes() final;
    void run_receiver_threads() final;
};

} // namespace rtp_receiver
} // namespace apps
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_APPS_RTP_RECEIVER_RTP_RECEIVER_H_ */
