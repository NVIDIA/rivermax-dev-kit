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

#ifndef RMAX_APPS_LIB_APPS_RMAX_XSTREAM_GENERIC_SENDER_H_
#define RMAX_APPS_LIB_APPS_RMAX_XSTREAM_GENERIC_SENDER_H_

#include <string>
#include <vector>
#include <memory>
#include <climits>

#include <rivermax_api.h>

#include "CLI/CLI.hpp"

#include "api/rmax_apps_lib_api.h"
#include "io_node/io_node.h"
#include "apps/rmax_base_app.h"

using namespace ral::lib::core;
using namespace ral::lib::services;
using namespace ral::io_node;

namespace ral
{
namespace apps
{
namespace rmax_xstream_generic_sender
{

/**
 * Application constants.
 */
constexpr const char* APP_DESCRIPTION = "NVIDIA Rivermax generic sender demo app ";
// TODO: Change application examples after adding unified rmax_xstream with sub commands interface.
constexpr const char* APP_EXAMPLES = \
    "\nExamples:\n"
    "  1. rmax_xstream_generic_sender --local-ip 1.2.3.4\n"
    "  2. rmax_xstream_generic_sender --local-ip 1.2.3.4 --dst-ip 1.2.5.6 --dst-port 56789\n"
    "  3. rmax_xstream_generic_sender --local-ip 1.2.3.4 --threads 10 --streams 20 --flows "
    "100 --internal-core 2 --application-core 2,4,6\n"
    "  4. rmax_xstream_generic_sender --local-ip 1.2.3.4 --streams 2 --flows 1000000 --chunks 2 "
    "--packets 4096 --payload-size 64\n"
    "  5. rmax_xstream_generic_sender --local-ip 1.2.3.4 --streams 2 --flows 10 --packets 4 "
    "--payload-size 512 --rate-bps 5000000 --rate-burst 1 -v\n";
/**
 * @brief: Generic Sender application.
 *
 * This is an example of usage application for Rivermax generic TX API.
 */
class GenericSenderApp : public RmaxBaseApp
{
private:
    /* Sender objects container */
    std::vector<std::unique_ptr<GenericSenderIONode>> m_senders;
    /* Stream per thread distribution */
    std::unordered_map<size_t,size_t> m_streams_per_thread;
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
     * @param [in] argc: Number of CLI arguments.
     * @param [in] argv: CLI arguments strings array.
     */
    GenericSenderApp(int argc, const char* argv[]);
    virtual ~GenericSenderApp();
    ReturnStatus run() override;
private:
    void add_cli_options() override;
    void post_cli_parse_initialization() override;
    ReturnStatus initialize_rivermax_resources() override;
    ReturnStatus cleanup_rivermax_resources() override;
    /**
     * @brief: Initializes network send flows.
     *
     * This method is responsible to initialize the send flows will be
     * used in the application. Those flows will be distributed in
     * @ref ral::apps::GenericSenderApp::distribute_work_for_threads
     * to the streams will be used in the application.
     * The application supports unicast and multicast UDPv4 send flows.
     *
     * @note:
     *    * Maximum number of multicast flows supported is @ref ral::lib::services::MAX_NUM_OF_TOTAL_FLOWS.
     *    * Maximum number of unicast flows supported is @ref ral::lib::services::MAX_PORT.
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
     * This method is responsible to initialize @ref ral::io_node::GenericSenderIONode objects to work.
     * It will initiate objects with the relevant parameters.
     * The objects initialized in this method, will be the contexts to the std::thread objects
     * will run in @ref ral::apps::RmaxBaseApp::run_threads method.
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
     * using @ref ral::lib::services::MemoryAllocator interface.
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
} // namespace ral

#endif /* RMAX_APPS_LIB_APPS_RMAX_XSTREAM_GENERIC_SENDER_H_ */
