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

#ifndef RDK_APPS_RMAX_XSTREAM_GENERIC_SENDER_RMAX_XSTREAM_GENERIC_SENDER_H_
#define RDK_APPS_RMAX_XSTREAM_GENERIC_SENDER_RMAX_XSTREAM_GENERIC_SENDER_H_

#include "rdk/apps/rmax_base_app.h"

using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
{
namespace apps
{
namespace rmax_xstream_generic_sender
{
/**
 * @brief: Configuration settings for Rivermax Generic Sender.
 */
struct GenericSenderSettings : AppSettings
{
public:
    void init_default_values() override;
};

/**
 * @brief: Validator for Rivermax Generic Sender settings.
 */
class GenericSenderSettingsValidator : public ISettingsValidator<GenericSenderSettings>
{
public:
     ReturnStatus validate(const std::shared_ptr<GenericSenderSettings>& settings) const override;
};

/**
 * @brief: CLI settings Builder for Rivermax Generic Sender.
 */
class GenericSenderCLISettingsBuilder : public CLISettingsBuilder<GenericSenderSettings>
{
public:
    /**
     * @brief: GenericSenderCLISettingsBuilder constructor.
     *
     * @param [in] argc: Number of CLI arguments.
     * @param [in] argv: CLI arguments strings array.
     * @param [in] app_description: Application description string for the CLI usage.
     * @param [in] app_examples: Application examples string for the CLI usage.
     */
    GenericSenderCLISettingsBuilder(int argc, const char** argv,
        const std::string& app_description,
        const std::string& app_examples,
        std::shared_ptr<ISettingsValidator<GenericSenderSettings>> validator) :
        CLISettingsBuilder<GenericSenderSettings>(argc, argv, app_description, app_examples, std::move(validator)) {}
    virtual ~GenericSenderCLISettingsBuilder() = default;
protected:
    ReturnStatus add_cli_options(std::shared_ptr<GenericSenderSettings>& settings) override;
};

using GenericSenderExternalSettingsBuilder = ExternalSettingsBuilder<GenericSenderSettings>;
/**
 * @brief: Generic Sender application.
 *
 * This is an example of usage application for Rivermax generic TX API.
 */
class GenericSenderApp : public RmaxBaseApp
{
private:
    /* Settings builder pointer */
    std::shared_ptr<ISettingsBuilder<GenericSenderSettings>> m_settings_builder;
    /* Application settings pointer */
    std::shared_ptr<GenericSenderSettings> m_generic_sender_settings;
    /* Sender objects container */
    std::vector<std::unique_ptr<GenericSenderIONode>> m_senders;
    /* Flow per stream distribution */
    std::unordered_map<int,size_t> m_flows_per_stream;
    /* Flow per thread distribution */
    std::unordered_map<size_t,size_t> m_flows_per_thread;
    /* Network send flows */
    std::vector<TwoTupleFlow> m_flows;
    /* Application memory blocks */
    std::vector<rmx_mem_region> m_mem_regions;
public:
    /**
     * @brief: GenericSenderApp class constructor.
     *
     * @param [in] settings_builder: Settings builder pointer.
     */
    GenericSenderApp(std::shared_ptr<ISettingsBuilder<GenericSenderSettings>> settings_builder);
    virtual ~GenericSenderApp() = default;
    ReturnStatus run() override;
    ReturnStatus initialize() override;
private:
    ReturnStatus initialize_app_settings() final;
    ReturnStatus post_load_settings() final;
    ReturnStatus cleanup_rivermax_resources() override;
    /**
     * @brief: Initializes network send flows.
     *
     * This method is responsible to initialize the send flows will be
     * used in the application. Those flows will be distributed in
     * @ref apps::GenericSenderApp::distribute_work_for_threads
     * to the streams will be used in the application.
     * The application supports unicast and multicast UDPv4 send flows.
     *
     * @note:
     *    * Maximum number of multicast flows supported is @ref services::MAX_NUM_OF_TOTAL_FLOWS.
     *    * Maximum number of unicast flows supported is @ref services::MAX_PORT.
     */
    void initialize_send_flows();
    /**
     * @brief: Distributes work for threads.
     *
     * This method is responsible to distribute work to threads, by
     * distributing number of streams per sender thread uniformly.
     * In future development, this can be extended to different
     * streams per thread distribution policies.
     */
    void distribute_work_for_threads();
    /**
     * @brief: Initializes sender threads.
     *
     * This method is responsible to initialize @ref io_node::GenericSenderIONode objects to work.
     * It will initiate objects with the relevant parameters.
     * The objects initialized in this method, will be the contexts to the std::thread objects
     * will run in @ref apps::RmaxBaseApp::run_threads method.
     */
    void initialize_sender_threads();
    /**
     * @brief: Returns the required memory length for the application.
     *
     * This method is responsible to calculate the memory required for the application.
     * It will do so by iterating it's senders and each sender's streams.
     *
     * @return: Required memory length.
     */
    size_t get_memory_length();
    /**
     * @brief: Allocates application memory.
     *
     * This method is responsible to allocate the required memory for the application
     * using @ref services::MemoryAllocator interface.
     * The allocation policy of the application is allocating one big memory
     * block and register it using @ref rmax_register_memory call.
     * This memory block will be distributed to the different components of the application.
     *
     * @return: Returns status of the operation.
     */
    ReturnStatus allocate_app_memory();
    /**
     * @brief: Distributes memory for senders.
     *
     * This method is responsible to distribute the memory allocated
     * by @ref allocate_app_memory to the senders of the application.
     *
     * @param [in] mem_block_index: The index of the memory block to use.
     */
    void distribute_memory_for_senders(const int mem_block_index);
};

} // namespace rmax_xstream_generic_sender
} // namespace apps
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_APPS_RMAX_XSTREAM_GENERIC_SENDER_RMAX_XSTREAM_GENERIC_SENDER_H_ */
