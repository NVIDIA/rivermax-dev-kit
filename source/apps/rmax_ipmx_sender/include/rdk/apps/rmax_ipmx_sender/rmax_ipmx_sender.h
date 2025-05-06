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

#ifndef RDK_APPS_RMAX_IPMX_SENDER_RMAX_IPMX_SENDER_H_
#define RDK_APPS_RMAX_IPMX_SENDER_RMAX_IPMX_SENDER_H_

#include "rdk/apps/rmax_base_app.h"

using namespace rivermax::dev_kit::apps;
using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
{
namespace apps
{
namespace rmax_ipmx_sender
{
constexpr size_t MIN_FRAMES_FOR_SIMULTANEOUS_TX_AND_FILLUP = 2;
/**
 * @brief: Configuration settings for Rivermax IPMX Sender.
 */
struct IPMXSenderSettings : AppSettings
{
public:
    void init_default_values() override;
};

/**
 * @brief: Validator for Rivermax IPMX Sender settings.
 */
class IPMXSenderSettingsValidator : public ISettingsValidator<IPMXSenderSettings>
{
public:
     ReturnStatus validate(const std::shared_ptr<IPMXSenderSettings>& settings) const override;
};

/**
 * @brief: CLI settings Builder for Rivermax IPMX Sender.
 */
class IPMXSenderCLISettingsBuilder : public CLISettingsBuilder<IPMXSenderSettings>
{
public:
    /**
     * @brief: IPMXSenderCLISettingsBuilder constructor.
     *
     * @param [in] argc: Number of CLI arguments.
     * @param [in] argv: CLI arguments strings array.
     * @param [in] app_description: Application description string for the CLI usage.
     * @param [in] app_examples: Application examples string for the CLI usage.
     */
    IPMXSenderCLISettingsBuilder(int argc, const char** argv,
        const std::string& app_description,
        const std::string& app_examples,
        std::shared_ptr<ISettingsValidator<IPMXSenderSettings>> validator) :
        CLISettingsBuilder<IPMXSenderSettings>(argc, argv, app_description, app_examples, std::move(validator)) {}
    virtual ~IPMXSenderCLISettingsBuilder() = default;
protected:
    ReturnStatus add_cli_options(std::shared_ptr<IPMXSenderSettings>& settings) override;
};

using IPMXSenderExternalSettingsBuilder = ExternalSettingsBuilder<IPMXSenderSettings>;

/**
 * @brief: IPMX Sender application.
 */
class IPMXSenderApp : public RmaxBaseApp
{
private:
    /* Settings builder pointer */
    std::shared_ptr<ISettingsBuilder<IPMXSenderSettings>> m_settings_builder;
    /* Application settings pointer */
    std::shared_ptr<IPMXSenderSettings> m_ipmx_sender_settings;
    std::vector<std::unique_ptr<IPMXSenderIONode>> m_senders;
    std::vector<TwoTupleFlow> m_stream_dst_addresses;
    rmx_device_iface m_device_interface;
    rmx_mem_region m_mem_region;
public:
    /**
     * @brief: IPMXSenderApp class constructor.
     *
     * @param [in] settings_builder: Settings builder pointer.
     */
    IPMXSenderApp(std::shared_ptr<ISettingsBuilder<IPMXSenderSettings>> settings_builder);
    virtual ~IPMXSenderApp() = default;
    ReturnStatus run() override;
    ReturnStatus initialize() override;
private:
    ReturnStatus initialize_app_settings() final;
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
     * @ref services::MemoryAllocator "memory-allocator".
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
     * Initializes @ref io_node::IPMXSenderIoNode "IPMX Sender IO Node" objects,
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
    /**
     * @brief: Checks if the given device has the specified address.
     *
     * @param [in] device: Network device to check.
     * @param [in] addr: IP address to check.
     *
     * @return: Return status of the operation.
     */
    static bool device_has_ip(const rmx_device* device, const in_addr& addr);
    /**
     * @brief: Obtains the MAC address of the given device.
     *
     * @param [in] device: Device interface to get the MAC address from.
     * @param [out] mac: MAC address in text form.
     *
     * @return: Return status of the operation.
     */
    static ReturnStatus read_device_mac_address(const rmx_device* device, std::string& mac);
    /**
     * @brief: Obtains the MAC address of the local interface used by the application.
     *
     * @param [out] mac: MAC address in text form.
     *
     * @return: Return status of the operation.
     */
    ReturnStatus read_local_mac_address(std::string& mac) const;
};

} // namespace rmax_ipmx_sender
} // namespace apps
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_APPS_RMAX_IPMX_SENDER_RMAX_IPMX_SENDER_H_ */
