/*
 * Copyright © 2017-2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#include <memory>

#include "rt_threads.h"

#include "rmax_apps_lib_facade.h"
#include "services/error_handling/error_handling.h"
#include "services/memory_management/memory_management.h"
#include "services/cli/options.h"

using namespace ral::lib::services;

ral::lib::RmaxAppsLibFacade::RmaxAppsLibFacade() :
    m_rmax_lib_initialized(false),
    m_cli_parser_manager(nullptr),
    m_mem_allocator(nullptr),
    m_signal_handler(nullptr)
{
}

ral::lib::RmaxAppsLibFacade::~RmaxAppsLibFacade()
{
    cleanup_rivermax();
}

std::shared_ptr<MemoryAllocator> ral::lib::RmaxAppsLibFacade::get_memory_allocator(
    AllocatorType type, std::shared_ptr<AppSettings> app_settings)
{
    if (!m_mem_allocator) {
        m_mem_allocator = MemoryAllocator::get_memory_allocator(type, app_settings);
    }

    return m_mem_allocator;
}

std::shared_ptr<CLIParserManager> ral::lib::RmaxAppsLibFacade::get_cli_parser_manager(
    const std::string& app_description, const std::string& app_examples,
    std::shared_ptr<AppSettings> app_settings)
{
    if (!m_cli_parser_manager) {
        m_cli_parser_manager = std::shared_ptr<CLIParserManager>(
            new CLIParserManager(app_description, app_examples, app_settings));
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

ReturnStatus ral::lib::RmaxAppsLibFacade::validate_rivermax_version() const
{
    std::stringstream app_version;
    app_version << RMAX_MAJOR_VERSION << "." << RMAX_MINOR_VERSION
        << "." << RMAX_PATCH_VERSION;

    std::stringstream version_header;
    version_header <<
        "###############################################\n" <<
        "## Rivermax SDK version:        " << rmax_get_version_string() << "\n" <<
        "## Application version:         " << app_version.str() << "\n" <<
        "###############################################\n";
    std::cout << version_header.str();

    return ReturnStatus::success;
}

ReturnStatus ral::lib::RmaxAppsLibFacade::initialize_rivermax(rmax_init_config& config)
{
    ReturnStatus rc = validate_rivermax_version();
    if (rc == ReturnStatus::rmax_version_incompatible) {
        return rc;
    }

    rmax_status_t status = rmax_init(&config);
    if (status != RMAX_OK) {
        std::cerr << "Failed to initialize Rivermax with status: " << status << std::endl;
        return ReturnStatus::failure;
    }

    m_rmax_lib_initialized = true;

    return ReturnStatus::success;
}

ReturnStatus ral::lib::RmaxAppsLibFacade::set_rivermax_clock(rmax_clock_t& clock)
{
    rmax_status_t status = rmax_set_clock(&clock);

    if (status != RMAX_OK) {
        std::cerr << "Failed to set Rivermax clock with status:" << status << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus ral::lib::RmaxAppsLibFacade::cleanup_rivermax() const
{
    if (!m_rmax_lib_initialized) {
        return ReturnStatus::success;
    }

    rmax_status_t status = rmax_cleanup();
    if (status != RMAX_OK) {
        std::cerr << "Failed to cleanup Rivermax with status:" << status << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}
