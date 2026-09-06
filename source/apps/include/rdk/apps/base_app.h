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

#ifndef RDK_APPS_BASE_APP_H_
#define RDK_APPS_BASE_APP_H_

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "rdk/facade.h"
#include "rdk/apps/memory_strategy_interface.h"
#include "rdk/io_node/io_node.h"
#include "rdk/services/services.h"
#include "rdk/services/utils/lifecycle.h"

using namespace rdk::io_node;
using namespace rdk::services;

namespace rdk
{
namespace apps
{
/**
 * @brief: Base calls for Rivermax Dev Kit application.
 *
 * This is a base class offers common operations for Rivermax Dev Kit application.
 * The user of this interface must implement it's pure virtual methods and
 * can override it's virtual methods.
 */
class BaseApp : public ILifecycle
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
    std::vector<sockaddr_in> m_local_addresses;
    /* NIC device interfaces */
    std::vector<rmx_device_iface> m_device_interfaces;
    /* Number of paths per stream */
    size_t m_num_paths_per_stream = 0;
    /* Header memory allocator */
    std::shared_ptr<MemoryAllocator> m_header_allocator;
    /* Payload memory allocator */
    std::shared_ptr<MemoryAllocator> m_payload_allocator;
    /* Auxiliary memory allocator */
    std::shared_ptr<MemoryAllocator> m_auxiliary_allocator;
    /* Application signal handler */
    std::shared_ptr<SignalHandler> m_signal_handler;
    /* Application GPU manager */
    std::shared_ptr<GPUManager> m_gpu_manager;
    /* Thread objects container */
    std::vector<std::thread> m_threads;
    /* Set when the whole app is asked to stop */
    std::atomic<bool> m_app_stop_requested{false};
    /* Whether the application's work is over. */
    std::atomic<bool> m_completed{false};
    /* Statistics reader */
    std::unique_ptr<StatisticsReader> m_stats_reader;
    /* Guards m_stats_reader. */
    std::mutex m_stats_reader_mutex;
    /* IO Node memory utilities */
    std::unique_ptr<::IONodeMemoryUtils> m_memory_utils;
    /* Memory strategy */
    std::unique_ptr<IMemoryStrategy> m_memory_strategy = nullptr;
public:
    virtual ~BaseApp();
    /**
     * @brief: Runs the application. Symmetric counterpart to @ref stop.
     *
     * This is the main entry point to the application. The initialization flow, using
     * @ref ::BaseApp::initialize call, should be run before calling to this method.
     * Blocks until every I/O node the application owns has finished.
     *
     * @return: Status of the operation.
     */
    ReturnStatus start() final
    {
        const ReturnStatus rc = do_work();
        // Only an application that gave up part way through starting its threads has any left.
        stop_io_nodes();
        stop_stats_reader();
        for (auto& thread : m_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        publish_completion();
        return rc;
    }
    /**
     * @brief: Requests the application to stop. Symmetric counterpart to @ref start.
     *
     * Call only after @ref start has begun. Non-blocking: stops every I/O node and the
     * statistics reader without waiting for them; @ref start returns once they have.
     */
    void stop() noexcept final
    {
        set_app_stop_requested();
        stop_stats_reader();
        stop_io_nodes();
    }
    /**
     * @brief: Requests one of the application's I/O nodes to stop.
     *
     * Non-blocking. Only that node stops; siblings and the application keep running.
     *
     * @param [in] id: I/O node's id, assigned by @ref initialize.
     *
     * @return: @ref ReturnStatus::success if @p id names an owned node, @ref ReturnStatus::failure otherwise.
     */
    virtual ReturnStatus stop_io_node(size_t id) = 0;
    /**
     * @brief: Waits for one of the application's I/O nodes to finish.
     *
     * Returns immediately if @p id names no owned node, or if that node has already finished.
     * Otherwise blocks for @p timeout, then reports whether it finished.
     *
     * @param [in] id: I/O node's id, assigned by @ref initialize.
     * @param [in] timeout: Maximum time to wait.
     *
     * @return: @ref ReturnStatus::success, @ref ReturnStatus::still_active, or @ref ReturnStatus::failure
     *          if @p id names no owned node.
     */
    virtual ReturnStatus await_io_node(size_t id, std::chrono::nanoseconds timeout) const = 0;
    /**
     * @brief: Waits for the application to finish. Bounded counterpart to joining the thread
     * that called @ref start.
     *
     * Returns immediately if @ref start has already returned. Otherwise blocks for @p timeout,
     * then reports whether it has.
     *
     * @param [in] timeout: Maximum time to wait.
     *
     * @return: @ref ReturnStatus::success if @ref start has returned;
     *          @ref ReturnStatus::still_active otherwise.
     */
    ReturnStatus await(std::chrono::nanoseconds timeout) const override;
    /**
     * @brief: Checks whether the application should stop.
     *
     * True once @ref stop has been called on this application, the process has received SIGINT,
     * or every I/O node has finished.
     *
     * @return: true if the application should stop.
     */
    bool should_stop() const
    {
        return m_app_stop_requested.load() || SignalHandler::get_received_signal() >= 0 ||
               all_io_nodes_stopped();
    }
    /**
     * @brief: Checks whether every I/O node the application owns has finished.
     *
     * @return: true if the application owns at least one node and all of them have finished.
     */
    bool all_io_nodes_stopped() const
    {
        const size_t num_of_nodes = num_io_nodes();
        if (num_of_nodes == 0) {
            return false;
        }
        for (size_t id = 0; id < num_of_nodes; ++id) {
            if (await_io_node(id, std::chrono::nanoseconds(0)) != ReturnStatus::success) {
                return false;
            }
        }
        return true;
    }
    /**
     * @brief: Returns the number of I/O nodes the application owns.
     *
     * Zero until @ref initialize has built the nodes.
     *
     * @return: Number of I/O nodes.
     */
    virtual size_t num_io_nodes() const = 0;
    /**
     * @brief: Checks whether the application was asked to stop.
     *
     * @return: true if @ref stop was called.
     */
    bool is_app_stop_requested() const noexcept { return m_app_stop_requested.load(); }
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
     * @brief: BaseApp constructor.
     *
     */
    BaseApp();
    /**
     * @brief: Initializes application settings.
     *
     * Use this method to initialize application settings.
     * Override it if you extend the AppSettings class.
     * It will be called as part of the @ref ::BaseApp::initialize process.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus initialize_app_settings() = 0;
    /**
     * @brief: Does post parsing initialization.
     *
     * Use this method to do any needed post loading application settings.
     * It will be called as part of the @ref ::BaseApp::initialize process.
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
     * There is no need to call @ref rmx_cleanup, it will be done implicitly.
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
     * @brief: Initializes the connection parameters.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus initialize_connection_parameters();
private:
    /**
     * @brief: The application's own work: runs its I/O nodes and waits for them.
     *
     * Private, so the only way to run an application is through @ref start.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus do_work() = 0;
    /**
     * @brief: Publishes the application's completion.
     */
    void publish_completion() noexcept { m_completed.store(true); }
protected:
    /**
     * @brief: Runs application threads.
     *
     * @param [in] io_nodes: A container of IONodes, it should follow STL containers interface.
     */
    template<typename T>
    void run_threads(T& io_nodes);
    /**
     * @brief: Requests one node in a container to stop, by its position.
     *
     * @param [in] io_nodes: A container of IIONode-derived elements, following the
     *                       STL container interface and supporting @c operator[].
     * @param [in] id: Position of the node in @p io_nodes.
     *
     * @return: @ref ReturnStatus::success if @p id names a node in @p io_nodes;
     *          @ref ReturnStatus::failure if it does not.
     */
    template<typename T>
    ReturnStatus stop_io_node(T& io_nodes, size_t id);
    /**
     * @brief: Requests every node in a container to stop.
     *
     * @param [in] io_nodes: A container of IIONode-derived elements, following the
     *                       STL container interface and supporting @c operator[].
     */
    template<typename T>
    void stop_io_nodes(T& io_nodes);
    /**
     * @brief: Waits for one node in a container to finish, by its position.
     *
     * @param [in] io_nodes: A container of IIONode-derived elements, following the
     *                       STL container interface and supporting @c operator[].
     * @param [in] id: Position of the node in @p io_nodes.
     * @param [in] timeout: Maximum time to wait.
     *
     * @return: Same two outcomes as @ref IIONode::await; additionally,
     *          @ref ReturnStatus::failure if @p id does not name a node in @p io_nodes.
     */
    template<typename T>
    ReturnStatus await_io_node(const T& io_nodes, size_t id, std::chrono::nanoseconds timeout) const;
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
     * @brief: Requests the statistics reader thread to stop, if one was started.
     *
     * Non-blocking. A no-op if @ref run_stats_reader was never called or the reader
     * was not started.
     */
    void stop_stats_reader() noexcept {
        std::lock_guard<std::mutex> lock(m_stats_reader_mutex);
        if (m_stats_reader) {
            m_stats_reader->stop();
        }
    }
    /**
     * @brief: Marks the app as asked to stop.
     *
     * Non-blocking, idempotent, thread-safe. Does not stop anything by itself.
     */
    void set_app_stop_requested() noexcept { m_app_stop_requested.store(true); }
    /**
     * @brief: Requests every I/O node the application owns to stop.
     *
     * Non-blocking; implement it with @ref stop_io_nodes over the application's own container.
     */
    virtual void stop_io_nodes() noexcept = 0;
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
void BaseApp::run_threads(T& io_nodes)
{
    const size_t first_io_node_thread = m_threads.size();
    for (auto& io_node : io_nodes) {
        m_threads.push_back(std::thread(std::ref(*io_node)));
    }

    for (size_t i = first_io_node_thread; i < m_threads.size(); ++i) {
        m_threads[i].join();
    }
    stop_stats_reader();
    for (auto& thread : m_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

template<typename T>
ReturnStatus BaseApp::stop_io_node(T& io_nodes, size_t id)
{
    if (id >= io_nodes.size()) {
        return ReturnStatus::failure;
    }
    io_nodes[id]->stop();
    return ReturnStatus::success;
}

template<typename T>
void BaseApp::stop_io_nodes(T& io_nodes)
{
    for (size_t id = 0; id < io_nodes.size(); id++) {
        stop_io_node(io_nodes, id);
    }
}

template<typename T>
ReturnStatus BaseApp::await_io_node(const T& io_nodes, size_t id, std::chrono::nanoseconds timeout) const
{
    if (id >= io_nodes.size()) {
        return ReturnStatus::failure;
    }
    return io_nodes[id]->await(timeout);
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
        ValidatorType validator;
        auto settings_builder = std::make_unique<SettingsBuilderType>(
            argc, argv, app_description + " " + rmx_get_version_string(), app_examples, validator);

        AppType app(std::move(settings_builder));
        ReturnStatus rc = app.initialize();
        if (rc == ReturnStatus::success_cli_help) {
            return EXIT_SUCCESS;
        } else if (rc != ReturnStatus::success) {
            std::cerr << "Application failed to initialize" << std::endl;
            return EXIT_FAILURE;
        }
        rc = app.start();
        if (rc != ReturnStatus::success) {
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
} // namespace rdk

#endif /* RDK_APPS_BASE_APP_H_ */
