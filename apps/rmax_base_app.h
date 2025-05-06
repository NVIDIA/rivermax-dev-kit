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

#ifndef RMAX_APPS_LIB_APPS_RMAX_BASE_APP_H_
#define RMAX_APPS_LIB_APPS_RMAX_BASE_APP_H_

#include <string>
#include <thread>

#include "api/rmax_apps_lib_api.h"

using namespace ral::lib::services;

namespace ral
{
namespace apps
{

constexpr int INVALID_CORE_NUMBER = -1;

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
    /* Rmax apps lib facade */
    ral::lib::RmaxAppsLibFacade m_rmax_apps_lib;
    /* Command line manager */
    std::shared_ptr<CLIParserManager> m_cli_parser_manager;
    /* Local NIC address */
    sockaddr_in m_local_address;
    /* Header memory allocator */
    std::shared_ptr<MemoryAllocator> m_header_allocator;
    /* Payload memory allocator */
    std::shared_ptr<MemoryAllocator> m_payload_allocator;
    /* Application signal handler */
    std::shared_ptr<SignalHandler> m_signal_handler;
    /* Thread objects container */
    std::vector<std::thread> m_threads;
    /* Statistics reader */
    std::unique_ptr<StatisticsReader> m_stats_reader;
public:
    /**
     * @brief: RmaxBaseApp constructor.
     *
     * @param [in] app_description: Application description string for the CLI usage.
     * @param [in] app_examples: Application examples string for the CLI usage.
     */
    RmaxBaseApp(const std::string& app_description, const std::string& app_examples);
    virtual  ~RmaxBaseApp();
    /**
     * @brief: Runs the application.
     *
     * This is the main entry point to the application.
     * The initialization flow, using @ref ral::apps::RmaxBaseApp::initialize call, should be run
     * before calling to this method.
     *
     * @retun: Status of the operation.
     */
    virtual ReturnStatus run() = 0;
protected:
    /**
     * @brief: Initializes application common default settings.
     *
     * The user of this interface can override function, in order to override
     * application specific default setting.
     */
    virtual void initialize_common_default_app_settings();
    /**
     * @brief: Adds CLI options and/or arguments to the parser.
     *
     * Use this method to add CLI options to the application, by using @ref m_cli_parser_manager->add_option.
     * It will be called as part of the @ref ral::apps::RmaxBaseApp::initialize process.
     */
    virtual void add_cli_options() {};
    /**
     * @brief: Does post CLI parsing initialization.
     *
     * Use this method to do any needed post CLI parsing application initialization.
     * It will be called as part of the @ref ral::apps::RmaxBaseApp::initialize process.
     */
    virtual void post_cli_parse_initialization() {};
    /**
     * @brief: Initializes memory allocators.
     *
     * @retun: Status of the operation.
     */
    virtual ReturnStatus initialize_memory_allocators();
    /**
     * @brief: Runs application initialization flow.
     *
     * Use this method to run application initialization flow using the other methods in this
     * interface.
     * It will eventually initialize Rivermax library.
     *
     * The user can override the proposed initialization flow.
     *
     * @param [in] argc: Number of CLI arguments.
     * @param [in] argv: CLI arguments strings array.
     *
     * @retun: Status of the operation, on success: @ref ral::lib:services::ReturnStatus::obj_init_success.
     */
    virtual ReturnStatus initialize(int argc, const char* argv[]);
    /**
     * @brief: Initializes Rivermax library resources.
     *
     * Use this method to initialize Rivermax library configuration. It should
     * use @ref ral::lib::RmaxAppsLibFacade::initialize_rivermax method do so.
     *
     * @retun: Status of the operation.
     */
    virtual ReturnStatus initialize_rivermax_resources() = 0;
    /**
     * @brief: Cleans up Rivermax library resources.
     *
     * Use this method to clean up any resources associated with Rivermax library.
     * It will be called implicitly on class destruction.
     *
     * There is no need to call @ref rmax_cleanup, it will be done implicitly.
     *
     * @retun: Status of the operation.
     */
    virtual ReturnStatus cleanup_rivermax_resources() { return ReturnStatus::success; };
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
     * @retun: Status of the operation.
     */
    virtual ReturnStatus initialize_connection_parameters();
    /**
     * @brief: Runs application threads.
     *
     * This method will run application IONodes as the context for @ref std::thread.
     * The IONode should override the operator () as it's worker method.
     *
     * @param [in] io_nodes: A container of IONodes, it should follow STL containers interface.
     */
    template<typename T>
    void run_threads(T& io_nodes);
    /**
     * @brief: Runs statistics reader thread.
     */
    void run_stats_reader();

    /**
     * @brief: Check if need not run the statistics reader.
     *
     * @retun: true if need to run the statistics reader.
     */
    bool is_run_stats_reader() {
        return (m_app_settings->statistics_reader_core != INVALID_CORE_NUMBER);
    }
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

} // namespace apps
} // namespace ral

#endif /* RMAX_APPS_LIB_APPS_RMAX_BASE_APP_H_ */
