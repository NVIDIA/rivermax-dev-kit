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

#ifndef RDK_FACADE_H_
#define RDK_FACADE_H_

#include <memory>
#include <mutex>
#include <string>

#include "rdk/services/utils/gpu_manager.h"
#include "rdk/services/memory_allocation/memory_allocator_interface.h"
#include "rdk/services/cli/cli_manager.h"
#include "rdk/services/error_handling/signal_handler.h"

using namespace rivermax::dev_kit;
using namespace rivermax::dev_kit::services;

namespace rivermax
{
namespace dev_kit
{
/**
 * @brief: Rivermax Dev Kit facade class.
 *
 * This class will be the facade of the library.
 * It will be responsible for library initialization and cleanup.
 * In addition, it will expose methods for library core and services objects creation.
 */
class RivermaxDevKitFacade
{
private:
    /**
     * Weak pointer that tracks initialization handles given to clients.
     * Using weak_ptr allows cleanup to occur when all client handles are released,
     * without keeping the resource alive due to the singleton itself.
     */
    std::weak_ptr<void> m_rivermax_lib_handle;
    /* Mutex to ensure thread safety */
    static std::mutex m_mutex;
    /* Command line manager */
    std::shared_ptr<CLIParserManager> m_cli_parser_manager;
    /* Application signal handler */
    std::shared_ptr<SignalHandler> m_signal_handler;
    /* Application GPU Manager */
    std::shared_ptr<GPUManager> m_gpu_manager;
    /**
     * @brief: RivermaxDevKitFacade constructor.
     */
    RivermaxDevKitFacade();
    RivermaxDevKitFacade(const RivermaxDevKitFacade&) = delete;
    RivermaxDevKitFacade& operator=(const RivermaxDevKitFacade&) = delete;
    ~RivermaxDevKitFacade() = default;
public:
    /**
     * @brief: Returns the singleton instance of RivermaxDevKitFacade.
     *
     * @return: Reference to the singleton instance.
     */
    static RivermaxDevKitFacade& get_instance();
    /**
     * @brief: Factory method for memory allocator class.
     *
     * @param [in] type: Memory allocator type.
     * @param [in] app_settings: Application settings.
     *
     * @return: Shared pointer to the memory allocator object.
     */
    std::shared_ptr<MemoryAllocator> get_memory_allocator(AllocatorType type,
        std::shared_ptr<AppSettings> app_settings) const;
    /**
     * @brief: Factory method for CLI parser manager class.
     *
     * @param [in] app_description: Application description string for the CLI usage.
     * @param [in] app_examples: Application examples string for the CLI usage.
     * @param [in] app_settings: Application settings.
     *
     * @return: Shared pointer to the CLI parser manager object.
     */
    std::shared_ptr<CLIParserManager> get_cli_parser_manager(
        const std::string& app_description, const std::string& app_examples,
        std::shared_ptr<AppSettings> app_settings);
    /**
     * @brief: Factory method for signal handler class.
     *
     * @param [in] register_default_handler: Determines whether to register default signal handler for SIGINT signal.
     *                                       Defaults to false.
     *
     * @note: The default handler for SIGINT signal is @ref SignalHandler::default_signal_handler.
     *
     * @return: Shared pointer to the signal handler object.
     */
    std::shared_ptr<SignalHandler> get_signal_handler(bool register_default_handler = false);
    /**
     * @brief: Factory method for GPU manager class.
     *
     * @return: Shared pointer to the GPU manager object.
     */
    std::shared_ptr<GPUManager> get_gpu_manager();
    /**
     * @brief: Initializes Rivermax library.
     *
     * @param [out] rivermax_lib_handle: A shared_ptr handle that keeps Rivermax initialized while held.
     * @param [in] cpu_affinity: CPU number to assign for running Rivermax internal threads.
     * @param [in] enable_signal_handling: Enable OS signal handling by Rivermax.
     *
     * @note: The returned handle must be kept alive by the client for as long as
     *        Rivermax functionality is needed. When all handles are released,
     *        Rivermax will be automatically cleaned up.
     *
     * @return: Status of the operation.
     */
    ReturnStatus initialize_rivermax(
        std::shared_ptr<void>& rivermax_lib_handle, int cpu, bool enable_signal_handling = true);
private:
    /**
     * @brief: Validates Rivermax library version.
     *
     * This function will validate the version and check for incompatibility.
     * This is part of library initialization.
     *
     * @return: Status of the operation.
     */
    ReturnStatus validate_rivermax_version() const;
    /**
     * @brief: Helper function for Rivermax initialization.
     *
     * @param [in] cpu_affinity: CPU number to assign for running Rivermax internal threads.
     * @param [in] enable_signal_handling: Enable OS signal handling by Rivermax.
     *
     * @return: Status of the operation.
     */
    ReturnStatus initialize_rivermax_helper(int cpu, bool enable_signal_handling = true);
    /**
     * @brief: Cleans up Rivermax library.
     *
     * It will be invoked on object destruction implicitly.
     *
     * @return: Status of the operation.
     */
    ReturnStatus cleanup_rivermax();
};

} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_FACADE_H_ */
