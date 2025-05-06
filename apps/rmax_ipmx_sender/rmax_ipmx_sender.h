/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef RMAX_APPS_LIB_APPS_RMAX_IPMX_SENDER_H_
#define RMAX_APPS_LIB_APPS_RMAX_IPMX_SENDER_H_

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
namespace rmax_ipmx_sender
{
/**
 * Application constants.
 */
constexpr const char* APP_DESCRIPTION = "NVIDIA Rivermax IPMX sender demo app ";

constexpr const char* APP_EXAMPLES = \
    "\nExamples:\n"
    "  1. rmax_ipmx_sender --local-ip 1.2.3.4 --stream-type 1080p60 -v --ptp\n"
    "  2. rmax_ipmx_sender --local-ip 1.2.3.4 --stream-type 1080p60 --dst-ip 234.5.6.7 --dst-port 2000\n"
    "  3. rmax_ipmx_sender --local-ip 1.2.3.4 --stream-type 1080p60 --streams 10\n"
    "  4. rmax_ipmx_sender --local-ip 1.2.3.4 --stream-type 1080p60 --threads 2 --streams 10 -a 1,2 -c 3\n";

constexpr size_t MIN_FRAMES_FOR_SIMULTANEOUS_TX_AND_FILLUP = 2;

/**
 * @brief: IPMX Sender application.
 */
class IpmxSenderApp : public RmaxBaseApp
{
private:
    std::vector<std::unique_ptr<IpmxSenderIONode>> m_senders;
    std::unordered_map<size_t,size_t> m_streams_per_thread;
    std::vector<TwoTupleFlow> m_stream_dst_addresses;
    rmx_device_iface m_device_interface;
    rmx_mem_region m_mem_region;
public:
    /**
     * @brief: IpmxSenderApp class constructor.
     *
     * @param [in] argc: Number of CLI arguments.
     * @param [in] argv: CLI arguments strings array.
     */
    IpmxSenderApp(int argc, const char* argv[]);
    virtual ~IpmxSenderApp();
    ReturnStatus run() override;
private:
    void initialize_common_default_app_settings() final;
    void add_cli_options() override;
    void post_cli_parse_initialization() override;
    ReturnStatus initialize_rivermax_resources() override;
    ReturnStatus initialize_connection_parameters() final;
    ReturnStatus set_rivermax_clock() override;
    /**
     * @brief: Initializes network send flows.
     *
     * Initializes the send-flows to be used in the application.
     * The application supports unicast and multicast UDPv4 send flows.
     */
    void initialize_send_flows();
    /**
     * @brief: Calculates the required memory length for the application.
     *
     * Calculates the memory size required by all the senders of the
     * application and their streams.
     *
     * @return: Required memory length.
     */
    size_t query_memory_size();
    /**
     * @brief: Allocates payload memory and aligns it to page size.
     *
     * Allocates memory from a special payload allocator (e.g. GPU-Direct memory),
     * if such an allocator is configured. Otherwise, it defaults to the general purpose
     * memory allocator.
     * @note The requested memory size is implicitly aligned to the minimal size unit
     *             of the selected Allocator.
          *
     * @param [in] size: Requested allocation size.
     *
     * @return: Pointer to allocated memory.
     */
    void* allocate_and_align_payload(size_t size);
    /**
     * @brief: Allocates application memory.
     *
     * Allocates the memory required by the application utilizing the chosen
     * @ref ral::lib::services::MemoryAllocator "memory-allocator".
     *
     * @note The preferred method is a bulk allocation of a single memory block,
     * which then is registered with a single memory key. Then the memory is
     * distributed among the components of the application.
     * It utilizes better the internal resources of the NIC.
     *
     * @return: Returns status of the operation.
     */
    ReturnStatus allocate_app_memory();
    /**
     * @brief: Distributes memory to senders.
     *
     * Distributes the memory allocated by @ref allocate_app_memory
     * to the senders of the application.
     */
    void distribute_memory_to_senders();
    /**
     * @brief: Assigns streams to the worker threads.
     *
     * Assign each stream to its designated thread.
     * Several streams are assigned to the same thread (almost evenly),
     * if the amount of the threads is smaller than of the streams.
     */
    void assign_streams_to_threads();
    /**
     * @brief: Initializes sender threads.
     *
     * Initializes @ref ral::io_node::IpmxSenderIoNode "IPMX Sender IO Node" objects,
     * based on the given parameters.
     * These objects serve as contexts for the threads that are designated for streams.
     */
    void initialize_sender_threads();
    /**
     * @brief: Initializes NIC device interface.
     *
     * @param [in] device_iface: Device interface to cionfigure.
     *
     * @return: Return status of the operation.
     */
    ReturnStatus init_device_iface(rmx_device_iface& device_iface);
};

} // namespace rmax_ipmx_sender
} // namespace apps
} // namespace ral

#endif /* RMAX_APPS_LIB_APPS_RMAX_IPMX_SENDER_H_ */
