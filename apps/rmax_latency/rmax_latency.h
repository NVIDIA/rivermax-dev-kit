/*
 * Copyright © 2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef RMAX_APPS_LIB_APPS_RMAX_LATENCY_H_
#define RMAX_APPS_LIB_APPS_RMAX_LATENCY_H_

#include <string>
#include <vector>
#include <memory>
#include <climits>
#include <unordered_set>

#include <rivermax_api.h>

#include "CLI/CLI.hpp"

#include "api/rmax_apps_lib_api.h"
#include "io_node/misc/latency_io_node.h"
#include "io_node/misc/generic_latency_io_node.h"
#include "io_node/misc/media_latency_io_node.h"
#include "apps/rmax_base_app.h"

using namespace ral::lib::core;
using namespace ral::lib::services;
using namespace ral::io_node;

namespace ral
{
namespace apps
{
namespace rmax_latency
{
/**
* @brief: Latency measurement mode.
*/
enum class LatencyMode {
    /**
     * Ping-pong mode.
     * Measuring end-to-end delivery delay and its parts for a single-packet transfer
     * using Generic API without Packet Pacing.
     */
    PingPong,
    /**
     * Frame latency mode.
     * Measuring end-to-end delivery delay and its parts for a transfer of
     * an arbitrary size video frame. Data is sent with maximal speed using
     * Generic API without Packet Pacing.
     */
    Frame,
    /**
     * Media mode.
     * Measuring packet latencies using Media API with PAcket Pacing.
     * In this mode delays are measured between actual packet transmit or receive time
     * and the time when the packet transmission was scheduled according to SMPTE 2110-21.
     */
    Media
};

/**
 * @brief: Latency measurement application.
 *
 * This is an application for measuring Rivermax data transfer latency and its components in various use cases.
 */
class LatencyApp : public RmaxBaseApp
{
private:
    static constexpr uint32_t DEFAULT_NUM_OF_PACKETS_IN_CHUNK = 1;
    /* Receive multicast IP */
    std::string m_receive_ip;
    uint16_t m_receive_port;
    /* Operate as client, else operate as server (default) */
    bool m_client;
    /* Disable processing of realtime completion timestamps */
    bool m_disable_ts = false;
    /* Disable percentile calculation */
    bool m_disable_percentile = false;
    /* Latency measurement mode */
    LatencyMode m_latency_mode;
    /* Transmitter-Receiver objects container */
    std::unique_ptr<LatencyIONode> m_io_node;
    /* Transmit memory region */
    rmx_mem_region m_tx_mreg;
    /* Receive memory header region */
    rmx_mem_region m_rx_header_mreg;
    /* Receive memory payload region */
    rmx_mem_region m_rx_payload_mreg;
    /* Total measuring interval */
    size_t m_measure_interval_sec;
    /* Rivermax NIC device */
    rmx_device_iface m_device_interface;
    /* CLI option to set packets in chunk */
    CLI::Option* m_packets_option;
public:
    /**
     * @brief: LatencyApp class constructor.
     *
     * @param [in] argc: Number of CLI arguments.
     * @param [in] argv: CLI arguments strings array.
     */
    LatencyApp(int argc, const char* argv[]);
    virtual ~LatencyApp() = default;
    ReturnStatus run() override;
private:
    void initialize_common_default_app_settings() final;
    void add_cli_options() final;
    ReturnStatus initialize_rivermax_resources() final;
    ReturnStatus initialize_connection_parameters() final;
    ReturnStatus set_rivermax_clock() final;
    /**
     * @brief: Initializes receiver threads.
     *
     * This method is responsible to initialize
     * @ref ral::io_node::LatencyIONode objects to work. It will initiate
     * objects with the relevant parameters. The objects initialized in this
     * method, will be the contexts to the std::thread objects will run in
     * @ref ral::apps::RmaxBaseApp::run_threads method.
     *
     * @return: Return status of the operation.
     */
    ReturnStatus initialize_threads();
    /**
     * @brief: Allocates application memory and registers it if requested.
     *
     * This method is responsible to allocate the required memory for the application
     * using @ref ral::lib::services::MemoryAllocator interface.
     * The allocation policy of the application is allocating one big memory
     * block for Transmit buffers and another for Receive buffers.
     * This method also registers the transmit memory.
     *
     * @return: Return status of the operation.
     */
    ReturnStatus allocate_app_memory();
    /**
     * @brief Unregister previously registered memory.
     *
     * Unregister memory using @ref rmax_deregister_memory.
     */
    void unregister_app_memory();
    /**
     * @brief: Distributes memory for receivers.
     *
     * This method is responsible to distribute the memory allocated
     * by @ref allocate_app_memory to the transmitter and receivers of the application.
     */
    void distribute_memory_for_streams();
    /**
     * @brief: Returns the required memory length for the application.
     *
     * This method is responsible to calculate the memory required for the application.
     * It will do so by iterating it's receivers and each receiver's streams.
     *
     * @param [out] hdr_mem_len: Required header memory length.
     * @param [out] pld_mem_len: Required payload memory length.
     *
     * @return: Return status of the operation.
     */
    ReturnStatus get_memory_length(size_t& hdr_mem_len, size_t& pld_mem_len);
    /**
     * @brief: Allocate header memory and align it to page size.
     *
     * This method allocates memory from a general purpose memory allocator.
     *
     * @param [in] size: Requested allocation size.
     *
     * @return: Pointer to allocated memory.
     */
    void* allocate_and_align_header(size_t size);
    /**
     * @brief: Allocate payload memory and align it to page size.
     *
     * This method allocates memory from a special payload allocator (e.g. GPU-Direct memory),
     * if such allocator is cofigured, and defaults to the general purpose memory allocator.
     * Allocator granularity is taken into account, and the requested memory size is aligned to
     * the minimal granularity.
     *
     * @param [in] size: Requested allocation size.
     *
     * @return: Pointer to allocated memory.
     */
    void* allocate_and_align_payload(size_t size);
    /**
     * @brief: Initialize NIC device interface.
     *
     * @param [in] device_iface: Device interface to cionfigure.
     *
     * @return: Return status of the operation.
     */
    ReturnStatus init_app_device_iface(rmx_device_iface& device_iface);
    /**
     * @brief: Check status of GPUDirect mode.
     *
     * @return: Return true if GPUDirect is enabled.
     */
    bool gpu_direct_enabled() {return (m_app_settings->gpu_id != INVALID_GPU_ID);}
    /**
     * @brief: Get the configured latency measurement mode.
     *
     * @return: Latency measurement mode.
     */
    LatencyMode get_latency_mode();
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

#endif /* RMAX_APPS_LIB_APPS_RMAX_LATENCY_H_ */
