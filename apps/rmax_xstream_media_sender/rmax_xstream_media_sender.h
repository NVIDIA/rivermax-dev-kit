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

#ifndef RMAX_APPS_LIB_APPS_RMAX_XSTREAM_MEDIA_SENDER_H_
#define RMAX_APPS_LIB_APPS_RMAX_XSTREAM_MEDIA_SENDER_H_

#include <string>
#include <vector>
#include <memory>
#include <climits>
#include <unordered_set>

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
namespace rmax_xstream_media_sender
{
/**
 * Application constants.
 */
constexpr const char* APP_DESCRIPTION = "NVIDIA Rivermax media sender demo app ";
// TODO: Change application examples after adding unified rmax_xstream with sub commands interface.
constexpr const char* APP_EXAMPLES = \
    "\nExamples:\n"
    "  1. rmax_xstream_media_sender --local-ip 1.2.3.4 --stream-type 1080p60 -v\n"
    "  2. rmax_xstream_media_sender --local-ip 1.2.3.4 --stream-type 1080p60 --dst-ip 234.5.6.7 --dst-port 2000\n"
    "  3. rmax_xstream_media_sender --local-ip 1.2.3.4 --stream-type 1080p60 --streams 10\n"
    "  4. rmax_xstream_media_sender --local-ip 1.2.3.4 --stream-type 1080p60 --threads 2 --streams 10 -a 1,2 -c 3\n";
/**
 * @brief: Media Sender application.
 *
 * This is an example of usage application for Rivermax media TX API.
 */
class MediaSenderApp : public RmaxBaseApp
{
private:
    /* Sender objects container */
    std::vector<std::unique_ptr<MediaSenderIONode>> m_senders;
    /* Stream per thread distribution */
    std::unordered_map<size_t,size_t> m_streams_per_thread;
    /* Network send flows */
    std::vector<TwoTupleFlow> m_flows;
public:
    /**
     * @brief: MediaSenderApp class constructor.
     *
     * @param [in] argc: Number of CLI arguments.
     * @param [in] argv: CLI arguments strings array.
     */
    MediaSenderApp(int argc, const char* argv[]);
    virtual ~MediaSenderApp();
    ReturnStatus run() override;
private:
    void add_cli_options() override;
    void post_cli_parse_initialization() override;
    ReturnStatus initialize_rivermax_resources() override;
    ReturnStatus set_rivermax_clock() override;
    /**
     * @brief: Initializes network send flows.
     *
     * This method is responsible to initialize the send flows will be
     * used in the application. Those flows will be distributed in
     * @ref ral::apps::MediaSenderApp::distribute_work_for_threads
     * to the streams will be used in the application.
     * The application supports unicast and multicast UDPv4 send flows.
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
     * This method is responsible to initialize @ref ral::io_node::MediaSenderIONode objects to work.
     * It will initiate objects with the relevant parameters.
     * The objects initialized in this method, will be the contexts to the std::thread objects
     * will run in @ref ral::apps::RmaxBaseApp::run_threads method.
     */
    void initialize_sender_threads();
    /**
     * @brief: Returns current time in nanoseconds.
     *
     * This method uses @ref std::chrono::system_clock to return the current TAI based time.
     *
     * @return: Current time in nanoseconds.
     */
    static uint64_t get_time_ns(void* context = nullptr);
};

} // namespace rmax_xstream_media_sender
} // namespace apps
} // namespace ral

#endif /* RMAX_APPS_LIB_APPS_RMAX_XSTREAM_MEDIA_SENDER_H_ */
