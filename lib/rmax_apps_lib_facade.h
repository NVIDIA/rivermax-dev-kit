/*
 * Copyright © 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef RMAX_APPS_LIB_LIB_RMAX_APPS_LIB_FACADE_H_
#define RMAX_APPS_LIB_LIB_RMAX_APPS_LIB_FACADE_H_

#include <functional>
#include <memory>
#include <vector>

#include "services/utils/defs.h"
#include "services/memory_management/memory_allocator_interface.h"
#include "services/error_handling/error_handling.h"

using namespace ral::lib::services;

namespace ral
{
namespace lib
{
/**
 * @brief: Rivermax application library facade class.
 *
 * This class will be the facade of the library.
 * It will be responsible for library initialization and cleanup.
 * In addition, it will expose methods for library core and services objects creation.
 */
class RmaxAppsLibFacade
{
private:
    bool m_rmax_lib_initialized;
    /* Command line manager */
    std::shared_ptr<CLIParserManager> m_cli_parser_manager;
    /* Application signal handler */
    std::shared_ptr<SignalHandler> m_signal_handler;
public:
    /**
     * @brief: RmaxAppsLibFacade constructor.
     */
    RmaxAppsLibFacade();
    ~RmaxAppsLibFacade();
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
     * @param [in] register_default_hanlder: Determines whether to register default signal handler for SIGINT signal.
     *                                       Defaults to false.
     *
     * @note: The default handler for SIGINT signal is @ref ral::lib::services::SignalHandler::default_signal_handler.
     *
     * @return: Shared pointer to the signal handler object.
     */
    std::shared_ptr<SignalHandler> get_signal_handler(bool register_default_hanlder = false);
    /**
     * @brief: Initializes Rivermax library.
     *
     * @param [in] cpu_affinity: CPU number to assign for running Rivermax internal threads.
     * @param [in] enable_signal_handling: Enable OS signal handling by Rivermax.
     *
     * @note: There is no need to run cleanup of Rivermax, it will be done implicitly.
     *
     * @return: Status of the operation.
     */
    ReturnStatus initialize_rivermax(int cpu, bool enable_signal_handling = true);
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
     * @brief: Cleans up Rivermax library.
     *
     * It will be invoked on object destruction implicitly.
     *
     * @return: Status of the operation.
     */
    ReturnStatus cleanup_rivermax() const;
};

} // namespace lib
} // namespace ral

#endif /* RMAX_APPS_LIB_LIB_RMAX_APPS_LIB_FACADE_H_ */
