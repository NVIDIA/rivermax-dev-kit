/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef RMAX_APPS_LIB_APPS_RMAX_RTP_RECEIVER_H_
#define RMAX_APPS_LIB_APPS_RMAX_RTP_RECEIVER_H_

#include <string>
#include <vector>
#include <memory>
#include <climits>
#include <unordered_set>

#include <rivermax_api.h>

#include "CLI/CLI.hpp"

#include "api/rmax_apps_lib_api.h"
#include "io_node/io_node.h"
#include "apps/rmax_receiver_base.h"

using namespace ral::lib::core;
using namespace ral::lib::services;
using namespace ral::io_node;

namespace ral
{
namespace apps
{
namespace rmax_rtp_receiver
{
/**
 * Application constants.
 */
constexpr const char* APP_DESCRIPTION = "NVIDIA Rivermax RTP receiver demo app";
constexpr const char* APP_EXAMPLES = \
    "\nExamples:\n"
    "  1. rmax_rtp_receiver --local-ip 1.2.3.4 --src-ip 6.7.8.9 --dst-ip 1.2.3.4 -p 50020 -v\n"
    "  2. rmax_rtp_receiver --local-ip 1.2.3.4 --src-ip 6.7.8.9 --dst-ip 1.2.3.4 -p 50020 --app-hdr-size 50 -v\n"
    "  3. rmax_rtp_receiver --local-ip 1.2.3.4 --src-ip 6.7.8.9 --dst-ip 239.1.1.1 -p 50020 --threads 2 --streams 10 -a 1,2 -i 3\n";

/**
 * @brief: RTP receiver application.
 *
 * This is an example of application that uses Rivermax API to receive multiple
 * RTP streams. It's intended to be as simple as possible so it doesn't include
 * redundancy or reordering logic. The number of dropped packets are calculated
 * using RTP sequence numbers. Depending on a command line parameters, the
 * application supports standard RTP sequence numbers or extended sequence
 * numbers as defined in SMPTE ST 2110-20.
 */
class RTPReceiverApp : public RmaxReceiverBaseApp
{
private:
    static constexpr uint32_t DEFAULT_NUM_OF_PACKETS_IN_CHUNK = 262144;

    /* Sender objects container */
    std::vector<std::unique_ptr<RTPReceiverIONode>> m_receivers;
    /* Network recv flows */
    std::vector<FourTupleFlow> m_flows;
    /* NIC device interface */
    rmx_device_iface m_device_iface {};
    /* Memory region for header memory */
    rmx_mem_region m_header_mem_region {};
    /* Memory region for payload memory */
    rmx_mem_region m_payload_mem_region {};
    /* Buffer sizes */
    std::vector<std::pair<size_t, size_t>> m_buffer_sizes;
    /* Is using extended sequence number? */
    bool m_is_extended_sequence_number = false;

public:
    /**
     * @brief: RTPReceiverApp class constructor.
     *
     * @param [in] argc: Number of CLI arguments.
     * @param [in] argv: CLI arguments strings array.
     */
    RTPReceiverApp(int argc, const char* argv[]);
    virtual ~RTPReceiverApp() = default;
private:
    void add_cli_options() final;
    ReturnStatus initialize_connection_parameters() final;
    /**
     * @brief: Initializes network receive flows.
     *
     * This method initializes the receive flows that will be used
     * in the application. These flows will be distributed
     * in @ref ral::apps::RTPReceiverApp::distribute_work_for_threads
     * between application threads.
     * The application supports unicast and multicast UDPv4 receive flows.
     */
    void configure_network_flows() final;
    /**
     * @brief: Initializes receiver I/O nodes.
     *
     * This method is responsible for initialization of
     * @ref ral::io_node::RTPReceiverIONode objects to work. It will initiate
     * objects with the relevant parameters. The objects initialized in this
     * method, will be the contexts to the std::thread objects will run in
     * @ref ral::apps::RmaxBaseApp::run_threads method.
     */
    void initialize_receive_io_nodes() final;
    /**
     * @brief: Registers previously allocated memory if requested.
     *
     * If @ref m_register_memory is set then this function registers
     * application memory using @ref rmax_register_memory.
     *
     * @return: Returns status of the operation.
     */
    ReturnStatus register_app_memory() final;
    /**
     * @brief: Unregister previously registered memory.
     *
     * Unregister memory using @ref rmax_deregister_memory.
     */
    void unregister_app_memory() final;
    /**
     * @brief: Distributes memory for receivers.
     *
     * This method is responsible for distributing the memory allocated
     * by @ref allocate_app_memory to the receivers of the application.
     */
    void distribute_memory_for_receivers() final;
    /**
     * @brief: Returns the memory size for all the receive streams.
     *
     * This method calculates the sum of memory sizes for all IONodes and their streams.
     *
     * @param [out] hdr_mem_size: Required header memory size.
     * @param [out] pld_mem_size: Required payload memory size.
     *
     * @return: Return status of the operation.
     */
    ReturnStatus get_total_streams_memory_size(size_t& hdr_mem_size, size_t& pld_mem_size) final;
    /**
     * @brief: Runs application threads.
     */
    void run_receiver_threads() final;
};

} // namespace rmax_rtp_receiver
} // namespace apps
} // namespace ral

#endif /* RMAX_APPS_LIB_APPS_RMAX_RTP_RECEIVER_H_ */
