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

#include <memory>
#include <string>
#include <iostream>

#include <rivermax_api.h>

#include "rdk/facade.h"
#include "rdk/services/error_handling/error_handling.h"
#include "rdk/services/memory_allocation/memory_allocation.h"
#include "rdk/services/cli/options.h"
#include "rdk/services/cpu/affinity/affinity.h"
#include "rdk/services/utils/gpu_manager.h"

using namespace rivermax::dev_kit::services;

std::mutex RivermaxDevKitFacade::m_mutex;

RivermaxDevKitFacade& RivermaxDevKitFacade::get_instance()
{
    static RivermaxDevKitFacade instance;
    return instance;
}

RivermaxDevKitFacade::RivermaxDevKitFacade() :
    m_cli_parser_manager(nullptr),
    m_signal_handler(nullptr)
{
}

std::shared_ptr<MemoryAllocator> RivermaxDevKitFacade::get_memory_allocator(
    AllocatorType type, std::shared_ptr<AppSettings> app_settings) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return MemoryAllocator::get_memory_allocator(type, std::move(app_settings));
}

std::shared_ptr<CLIParserManager> RivermaxDevKitFacade::get_cli_parser_manager(
    const std::string& app_description, const std::string& app_examples,
    std::shared_ptr<AppSettings> app_settings)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_cli_parser_manager) {
        m_cli_parser_manager = std::make_shared<CLIParserManager>(app_description, app_examples, std::move(app_settings));
    }

    return m_cli_parser_manager;
}

std::shared_ptr<SignalHandler> RivermaxDevKitFacade::get_signal_handler(bool register_default_handler)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_signal_handler) {
        m_signal_handler = std::make_shared<SignalHandler>(register_default_handler);
    }

    return m_signal_handler;
}

std::shared_ptr<GPUManager> RivermaxDevKitFacade::get_gpu_manager()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_gpu_manager) {
        m_gpu_manager = std::make_shared<GPUManager>();
    }

    return m_gpu_manager;
}

ReturnStatus RivermaxDevKitFacade::validate_rivermax_version() const
{
    std::stringstream app_version;
    app_version << RMX_VERSION_MAJOR << "." << RMX_VERSION_MINOR
        << "." << RMX_VERSION_PATCH;

    std::stringstream version_header;
    version_header <<
        "###############################################\n" <<
        "## Rivermax SDK version:        " << rmx_get_version_string() << "\n" <<
        "## Application version:         " << app_version.str() << "\n" <<
        "###############################################\n";
    std::cout << version_header.str();

    return ReturnStatus::success;
}

ReturnStatus RivermaxDevKitFacade::initialize_rivermax(
    std::shared_ptr<void>& rivermax_lib_handle, int cpu, bool enable_signal_handling)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    // Try to get an existing handle if available
    rivermax_lib_handle = m_rivermax_lib_handle.lock();

    if (!rivermax_lib_handle) {
        // No valid handle exists, need to initialize
        auto rc = initialize_rivermax_helper(cpu, enable_signal_handling);
        if (rc != ReturnStatus::success) {
            return rc;
        }

        // Create a new handle with custom deleter
        static int dummy;
        rivermax_lib_handle = std::shared_ptr<void>(
            static_cast<void*>(&dummy),
            [this](void*) {
                std::lock_guard<std::mutex> lock(m_mutex);
                cleanup_rivermax();
            }
        );
        // Store a weak reference to the handle
        m_rivermax_lib_handle = rivermax_lib_handle;
    }

    return ReturnStatus::success;
}

ReturnStatus RivermaxDevKitFacade::initialize_rivermax_helper(int cpu, bool enable_signal_handling)
{
    ReturnStatus rc = validate_rivermax_version();
    if (rc == ReturnStatus::rmax_version_incompatible) {
        std::cerr << "Incompatible Rivermax version." << std::endl;
        return rc;
    }

    auto is_success = set_rivermax_thread_cpu_affinity(cpu);
    if (!is_success) {
        std::cerr << "Failed to set Rivermax thread CPU affinity." << std::endl;
        return ReturnStatus::failure;
    }

    rmx_status status;

    if (enable_signal_handling) {
        status = rmx_enable_system_signal_handling();
        if (status != RMX_OK) {
            std::cerr << "Failed to enable signal handling with status: " << status << std::endl;
            return ReturnStatus::failure;
        }
    }

    status = rmx_init();

    if (status != RMX_OK) {
        std::cerr << "Failed to initialize Rivermax with status: " << status << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus RivermaxDevKitFacade::cleanup_rivermax()
{
    rmx_status status = rmx_cleanup();
    if (status != RMX_OK) {
        std::cerr << "Failed to cleanup Rivermax with status:" << status << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}
