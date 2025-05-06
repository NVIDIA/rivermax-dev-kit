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

#ifndef RDK_APPS_RMAX_BASE_APP_H_
#define RDK_APPS_RMAX_BASE_APP_H_

#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <unordered_map>

#include "rdk/facade.h"
#include "rdk/apps/memory_strategy_interface.h"
#include "rdk/io_node/io_node.h"
#include "rdk/services/services.h"

using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;

namespace rivermax
{
namespace dev_kit
{
namespace apps
{
/**
 * @brief: Base calls for Rivermax application.
 *
 * This is a base class offers common operations for Rivermax application.
 * The user of this interface must implement it's pure virtual methods and
 * can override it's virtual methods.
 */
class RmaxBaseApp
{
protected:
    /* Indicator on whether the object created correctly */
    ReturnStatus m_obj_init_status;
    /* Application settings pointer */
    std::shared_ptr<AppSettings> m_app_settings;
    /* Rivermax Dev Kit facade */
    RivermaxDevKitFacade& m_rivermax_dev_kit;
    /* Handle to Rivermax library */
    std::shared_ptr<void> m_rivermax_lib_handle = nullptr;
    /* Stream per thread distribution */
    std::unordered_map<size_t, size_t> m_streams_per_thread;
    /* Local NIC address */
    sockaddr_in m_local_address;
    /* Header memory allocator */
    std::shared_ptr<MemoryAllocator> m_header_allocator;
    /* Payload memory allocator */
    std::shared_ptr<MemoryAllocator> m_payload_allocator;
    /* Application signal handler */
    std::shared_ptr<SignalHandler> m_signal_handler;
    /* Application GPU manager */
    std::shared_ptr<GPUManager> m_gpu_manager;
    /* Thread objects container */
    std::vector<std::thread> m_threads;
    /* Statistics reader */
    std::unique_ptr<StatisticsReader> m_stats_reader;
    /* IO Node memory utilities */
    std::unique_ptr<::IONodeMemoryUtils> m_memory_utils;
    /* Memory strategy */
    std::unique_ptr<IMemoryStrategy> m_memory_strategy = nullptr;
public:
    virtual ~RmaxBaseApp();
    /**
     * @brief: Runs the application.
     *
     * This is the main entry point to the application.
     * The initialization flow, using @ref ::RmaxBaseApp::initialize call, should be run
     * before calling to this method.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus run() = 0;
    /**
     * @brief: Runs application initialization flow.
     *
     * Use this method to run application initialization flow using the other methods in this
     * interface. It will eventually initialize Rivermax library.
     *
     * The user can override the proposed initialization flow.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus initialize();
    /**
     * @brief: Returns application settings.
     *
     * If the application was initialized successfully, this method will set the
     * provided reference to the application settings and return success.
     * Otherwise, it will return a failure status.
     *
     * @param [out] settings: A reference to store the application settings.
     *
     * @return: Status of the operation.
     */
    ReturnStatus get_app_settings(const AppSettings*& settings) const;
protected:
    /**
     * @brief: RmaxBaseApp constructor.
     *
     */
    RmaxBaseApp();
    /**
     * @brief: Initializes application settings.
     *
     * Use this method to initialize application settings.
     * Override it if you extend the AppSettings class.
     * It will be called as part of the @ref ::RmaxBaseApp::initialize process.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus initialize_app_settings() = 0;
    /**
     * @brief: Does post parsing initialization.
     *
     * Use this method to do any needed post loading application settings.
     * It will be called as part of the @ref ::RmaxBaseApp::initialize process.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus post_load_settings() { return ReturnStatus::success; };
    /**
     * @brief: Initializes memory allocators.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus initialize_memory_allocators();
    /**
     * @brief: Initializes GPU manager.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus initialize_gpu_manager();
    /**
     * @brief: Initializes Rivermax library resources.
     *
     * Use this method to initialize Rivermax library configuration. It should
     * use @ref RivermaxDevKitFacade::initialize_rivermax method do so.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus initialize_rivermax_resources();
    /**
     * @brief: Configures memory layout.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus configure_memory_layout();
    /**
     * @brief: Initializes memory strategy.
     *
     * @note: TODO: When all application will switch to memory strategy utilization, this method should be pure virtual.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus initialize_memory_strategy() { return ReturnStatus::success; }
    /**
     * @brief: Cleans up Rivermax library resources.
     *
     * Use this method to clean up any resources associated with Rivermax library.
     * It will be called implicitly on class destruction.
     *
     * There is no need to call @ref rmax_cleanup, it will be done implicitly.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus cleanup_rivermax_resources();
    /**
     * @brief: Sets Rivermax clock.
     *
     * Use this method to set Rivermax clock.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus set_rivermax_clock();
    /**
     * @brief: Initializes the local NIC address.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus initialize_connection_parameters();
    /**
     * @brief: Runs application threads.
     *
     * This method will run application IONodes as the context for @ref std::thread.
     * The IO Node should override the operator () as it's worker method.
     *
     * @param [in] io_nodes: A container of IONodes, it should follow STL containers interface.
     */
    template<typename T>
    void run_threads(T& io_nodes);
    /**
     * @brief: Runs statistics reader thread.
     *
     * @return: Status of the operation.
     */
    ReturnStatus run_stats_reader();
    /**
     * @brief: Check if need not run the statistics reader.
     *
     * @return: true if need to run the statistics reader.
     */
    bool is_run_stats_reader() {
        return (m_app_settings->enable_statistics_reader);
    }
    /**
     * @brief: Check status of GPUDirect mode.
     *
     * @return: Return true if GPUDirect is enabled.
     */
    bool gpu_direct_enabled() { return (m_app_settings->gpu_id != INVALID_GPU_ID); }
    /**
     * @brief: Returns the internal stream index for a given external stream index.
     *
     * This function calculates the internal thread index and internal stream index
     * for a given external stream index based on the number of threads and streams per thread.
     *
     * @param [in] external_stream_index: The external stream index.
     * @param [out] thread_index: The calculated thread index.
     * @param [out] internal_stream_index: The calculated internal stream index.
     *
     * @return: Status of the operation.
     */
    ReturnStatus find_internal_stream_index(
        size_t external_stream_index, size_t& thread_index, size_t& internal_stream_index);
};

template<typename T>
void RmaxBaseApp::run_threads(T& io_nodes)
{
    for (auto& io_node : io_nodes) {
        m_threads.push_back(std::thread(std::ref(*io_node)));
    }

    for (auto& thread : m_threads) {
        thread.join();
    }
}

/**
 * @brief: Common templated main function for applications.
 *
 * This function initializes the settings builder, creates the application instance,
 * and runs the application.
 *
 * @tparam AppType: The type of the application.
 * @tparam SettingsBuilderType: The type of the settings builder.
 * @tparam ValidatorType: The type of the settings validator.
 * @param [in] argc: Number of command line arguments.
 * @param [in] argv: Array of command line arguments.
 * @param [in] app_description: Description of the application.
 * @param [in] app_examples: Examples of how to use the application.
 *
 * @return: Exit status of the application.
 */
template <typename AppType, typename SettingsBuilderType, typename ValidatorType>
int common_cli_main(int argc, const char* argv[],
    const std::string& app_description, const std::string& app_examples)
{
    try {
        auto validator = std::make_shared<ValidatorType>();
        auto settings_builder = std::make_shared<SettingsBuilderType>(
            argc, argv, app_description + rmx_get_version_string(), app_examples, std::move(validator));

        AppType app(std::move(settings_builder));
        ReturnStatus rc = app.initialize();
        if (rc == ReturnStatus::success_cli_help) {
            return EXIT_SUCCESS;
        } else if (rc != ReturnStatus::obj_init_success) {
            std::cerr << "Application failed to initialize" << std::endl;
            return EXIT_FAILURE;
        }
        rc = app.run();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Application failed to run" << std::endl;
            return EXIT_FAILURE;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

} // namespace apps
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_APPS_RMAX_BASE_APP_H_ */
