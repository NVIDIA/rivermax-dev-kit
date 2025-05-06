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

#ifndef RDK_APPS_RMAX_LATENCY_RMAX_LATENCY_H_
#define RDK_APPS_RMAX_LATENCY_RMAX_LATENCY_H_

#include "rdk/apps/rmax_base_app.h"

using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
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
 * @brief: Configuration settings for Latency App.
 */
struct LatencySettings : AppSettings
{
public:
    static constexpr const char* DEFAULT_SEND_IP = "224.1.1.1";
    static constexpr uint16_t DEFAULT_SEND_PORT = 2000;
    static constexpr const char* DEFAULT_RECEIVE_IP = "224.1.1.2";
    static constexpr uint16_t DEFAULT_RECEIVE_PORT = 2000;
    static constexpr size_t DEFAULT_SLEEP_USEC = 10;
    static constexpr size_t DEFAULT_MEASURE_SEC = 10;
    static constexpr uint32_t DEFAULT_NUM_OF_PACKETS_IN_CHUNK = 1;

    void init_default_values() override;

    /* Receive multicast IP */
    std::string receive_ip;
    /* Receive port */
    uint16_t receive_port;
    /* Operate as client, else operate as server (default) */
    bool client;
    /* Disable processing of realtime completion timestamps */
    bool disable_ts = false;
    /* Disable percentile calculation */
    bool disable_percentile = false;
    /* Latency measurement mode */
    LatencyMode latency_mode;
    /* Total measuring interval */
    size_t measure_interval_sec;
};

/**
 * @brief: Validator for Latency App settings.
 */
class LatencySettingsValidator : public ISettingsValidator<LatencySettings>
{
public:
     ReturnStatus validate(const std::shared_ptr<LatencySettings>& settings) const override;
};

/**
 * @brief: CLI settings Builder for Latency App.
 */
class LatencyCLISettingsBuilder : public CLISettingsBuilder<LatencySettings>
{
public:
    static constexpr int USECS_IN_SECOND = 1000000;
    /**
     * @brief: LatencyCLISettingsBuilder constructor.
     *
     * @param [in] argc: Number of CLI arguments.
     * @param [in] argv: CLI arguments strings array.
     * @param [in] app_description: Application description string for the CLI usage.
     * @param [in] app_examples: Application examples string for the CLI usage.
     */
    LatencyCLISettingsBuilder(int argc, const char** argv,
        const std::string& app_description,
        const std::string& app_examples,
        std::shared_ptr<ISettingsValidator<LatencySettings>> validator) :
        CLISettingsBuilder<LatencySettings>(argc, argv, app_description, app_examples, std::move(validator)) {}
    virtual ~LatencyCLISettingsBuilder() = default;
protected:
    ReturnStatus add_cli_options(std::shared_ptr<LatencySettings>& settings) override;
};

using LatencyExternalSettingsBuilder = ExternalSettingsBuilder<LatencySettings>;

/**
 * @brief: Latency measurement application.
 *
 * This is an application for measuring Rivermax data transfer latency and its components in various use cases.
 */
class LatencyApp : public RmaxBaseApp
{
private:
    /* Settings builder pointer */
    std::shared_ptr<ISettingsBuilder<LatencySettings>> m_settings_builder;
    /* Application settings pointer */
    std::shared_ptr<LatencySettings> m_latency_settings;
    /* Transmitter-Receiver objects container */
    std::unique_ptr<LatencyIONode> m_io_node;
    /* Transmit header memory region */
    rmx_mem_region m_tx_header_mreg;
    /* Transmit header memory registration status */
    bool m_is_tx_header_mreg_registered;
    /* Transmit payload memory region */
    rmx_mem_region m_tx_payload_mreg;
    /* Transmit payload memory registration status */
    bool m_is_tx_payload_mreg_registered;
    /* Receive header memory region */
    rmx_mem_region m_rx_header_mreg;
    /* Receive payload memory region */
    rmx_mem_region m_rx_payload_mreg;
    /* Rivermax NIC device */
    rmx_device_iface m_device_interface;
public:
    /**
     * @brief: LatencyApp class constructor.
     *
     * @param [in] settings_builder: Settings builder pointer.
     */
    LatencyApp(std::shared_ptr<ISettingsBuilder<LatencySettings>> settings_builder);
    virtual ~LatencyApp() = default;
    ReturnStatus run() override;
    ReturnStatus initialize() override;
private:
    ReturnStatus initialize_app_settings() final;
    ReturnStatus initialize_connection_parameters() final;
    ReturnStatus set_rivermax_clock() final;
    /**
     * @brief: Initializes receiver threads.
     *
     * This method is responsible to initialize
     * @ref io_node::LatencyIONode objects to work. It will initiate
     * objects with the relevant parameters. The objects initialized in this
     * method, will be the contexts to the std::thread objects will run in
     * @ref apps::RmaxBaseApp::run_threads method.
     *
     * @return: Return status of the operation.
     */
    ReturnStatus initialize_threads();
    /**
     * @brief: Allocates application memory and registers it if requested.
     *
     * This method is responsible to allocate the required memory for the application
     * using @ref services::MemoryAllocator interface.
     * The allocation policy of the application is allocating one big memory
     * block for Transmit buffers and another for Receive buffers.
     * This method also registers the transmit memory.
     *
     * @return: Return status of the operation.
     */
    ReturnStatus allocate_app_memory();
    /**
     * @brief: Unregister previously registered memory.
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
     * @brief: Returns the configured latency measurement mode.
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
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_APPS_RMAX_LATENCY_RMAX_LATENCY_H_ */
