/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <memory>
#include <thread>
#include <vector>
#include <cstdint>

#include "rmax_apps_lib_facade.h"
#include "services/error_handling/error_handling.h"
#include "services/memory_management/memory_management.h"
#include "services/cli/options.h"
#include "services/utils/cpu.h"
#include "services/utils/gpu_manager.h"

using namespace ral::lib::services;

ral::lib::RmaxAppsLibFacade::RmaxAppsLibFacade() :
    m_rmax_lib_initialized(false),
    m_cli_parser_manager(nullptr),
    m_signal_handler(nullptr)
{
}

ral::lib::RmaxAppsLibFacade::~RmaxAppsLibFacade()
{
    cleanup_rivermax();
}

std::shared_ptr<MemoryAllocator> ral::lib::RmaxAppsLibFacade::get_memory_allocator(
    AllocatorType type, std::shared_ptr<AppSettings> app_settings) const
{
    return MemoryAllocator::get_memory_allocator(type, std::move(app_settings));
}

std::shared_ptr<CLIParserManager> ral::lib::RmaxAppsLibFacade::get_cli_parser_manager(
    const std::string& app_description, const std::string& app_examples,
    std::shared_ptr<AppSettings> app_settings)
{
    if (!m_cli_parser_manager) {
        m_cli_parser_manager = std::shared_ptr<CLIParserManager>(
            new CLIParserManager(app_description, app_examples, std::move(app_settings)));
    }

    return m_cli_parser_manager;
}

std::shared_ptr<SignalHandler> ral::lib::RmaxAppsLibFacade::get_signal_handler(bool register_default_hanlder)
{
    if (!m_signal_handler) {
        m_signal_handler = std::shared_ptr<SignalHandler>(new SignalHandler(register_default_hanlder));
    }

    return m_signal_handler;
}

std::shared_ptr<GPUManager> ral::lib::RmaxAppsLibFacade::get_gpu_manager()
{
    if (!m_gpu_manager) {
        m_gpu_manager = std::make_shared<GPUManager>();
    }

    return m_gpu_manager;
}

ReturnStatus ral::lib::RmaxAppsLibFacade::validate_rivermax_version() const
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

ReturnStatus ral::lib::RmaxAppsLibFacade::initialize_rivermax(int cpu, bool enable_signal_handling)
{
    ReturnStatus rc = validate_rivermax_version();
    if (rc == ReturnStatus::rmax_version_incompatible) {
        return rc;
    }

    rc = set_rivermax_thread_cpu_affinity(cpu);
    if (rc != ReturnStatus::success) {
        return rc;
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
    } else {
        m_rmax_lib_initialized = true;
    }

    return ReturnStatus::success;
}

ReturnStatus ral::lib::RmaxAppsLibFacade::cleanup_rivermax() const
{
    if (!m_rmax_lib_initialized) {
        return ReturnStatus::success;
    }

    rmx_status status = rmx_cleanup();
    if (status != RMX_OK) {
        std::cerr << "Failed to cleanup Rivermax with status:" << status << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}
