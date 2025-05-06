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

#include "rt_threads.h"

#include "rdk/apps/rmax_ipmx_sender/rmax_ipmx_sender.h"
#include "rdk/services/utils/defs.h"
#include "rdk/services/utils/clock.h"
#include "rdk/services/media/media.h"

using namespace rivermax::dev_kit::apps::rmax_ipmx_sender;

void IPMXSenderSettings::init_default_values()
{
    AppSettings::init_default_values();
    media.frames_fields_in_mem_block = MIN_FRAMES_FOR_SIMULTANEOUS_TX_AND_FILLUP;
    ref_clk_is_ptp = false;
    app_memory_alloc = true;
    register_memory = true;
}

ReturnStatus IPMXSenderSettingsValidator::validate(const std::shared_ptr<IPMXSenderSettings>& settings) const
{
    ReturnStatus rc = ValidatorUtils::validate_ip4_address(settings->local_ip);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_ip4_address(settings->destination_ip);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_ip4_port(settings->destination_port);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_core(settings->internal_thread_core);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_core(settings->app_threads_cores);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_core(settings->statistics_reader_core);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    if (settings->register_memory && !settings->app_memory_alloc) {
        std::cerr << "Register memory option is supported only with application memory allocation" << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

ReturnStatus IPMXSenderCLISettingsBuilder::add_cli_options(std::shared_ptr<IPMXSenderSettings>& settings)
{
    if (m_cli_parser_manager == nullptr) {
        std::cerr << "CLI parser manager is not initialized" << std::endl;
        return ReturnStatus::failure;
    }
    m_cli_parser_manager->add_option(CLIOptStr::LOCAL_IP);
    m_cli_parser_manager->add_option(CLIOptStr::DST_IP);
    m_cli_parser_manager->add_option(CLIOptStr::DST_PORT);
    m_cli_parser_manager->add_option(CLIOptStr::THREADS);
    m_cli_parser_manager->add_option(CLIOptStr::STREAMS)->check(
        StreamToThreadsValidator(settings->num_of_threads));
    m_cli_parser_manager->add_option(CLIOptStr::VERBOSE);
    m_cli_parser_manager->add_option(CLIOptStr::INTERNAL_CORE);
    m_cli_parser_manager->add_option(CLIOptStr::APPLICATION_CORE);
    m_cli_parser_manager->add_option(CLIOptStr::SLEEP);
    m_cli_parser_manager->add_option(CLIOptStr::ALLOCATOR_TYPE);
    auto parser = m_cli_parser_manager->get_parser();
    m_cli_parser_manager->add_option(CLIOptStr::VIDEO_RESOLUTION)
        ->group(CLIGroupStr::VIDEO_FORMAT_OPTIONS);
    m_cli_parser_manager->add_option(CLIOptStr::VIDEO_FRAME_RATE)
        ->group(CLIGroupStr::VIDEO_FORMAT_OPTIONS);
    // TODO: move PTP flag to App
    parser->add_flag("--ptp", settings->ref_clk_is_ptp,
                     "Use NIC RTC as a PTP-synchronized Common Reference clock");
    auto stats_enabled = m_cli_parser_manager->add_option(CLIOptStr::ENABLE_STATS_READER);
    auto stats_core = m_cli_parser_manager->add_option(CLIOptStr::STATS_CORE)->needs(stats_enabled);
    m_cli_parser_manager->add_option(CLIOptStr::STATS_SESSION_ID)->needs(stats_enabled);
    return ReturnStatus::success;
}

IPMXSenderApp::IPMXSenderApp(std::shared_ptr<ISettingsBuilder<IPMXSenderSettings>> settings_builder) :
    RmaxBaseApp(),
    m_settings_builder(std::move(settings_builder)),
    m_device_interface{},
    m_mem_region{nullptr, 0, 0}
{
}

ReturnStatus IPMXSenderApp::initialize_app_settings()
{
    if (m_settings_builder == nullptr) {
        std::cerr << "Settings builder is not initialized" << std::endl;
        return ReturnStatus::failure;
    }
    m_ipmx_sender_settings = std::make_shared<IPMXSenderSettings>();
    ReturnStatus rc = m_settings_builder->build(m_ipmx_sender_settings);
    if (rc == ReturnStatus::success) {
        m_app_settings = m_ipmx_sender_settings;
        return ReturnStatus::success;
    }
    if (rc != ReturnStatus::success_cli_help) {
        std::cerr << "Failed to build settings" << std::endl;
    }
    m_obj_init_status = rc;
    return rc;
}

ReturnStatus IPMXSenderApp::initialize()
{
    ReturnStatus rc  = RmaxBaseApp::initialize();

    if (rc != ReturnStatus::obj_init_success) {
        return m_obj_init_status;
    }

    try {
        assign_streams_to_threads();
        initialize_send_flows();
        initialize_sender_threads();
        rc = allocate_app_memory();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to allocate the memory required for the application" << std::endl;
            return rc;
        }
        distribute_memory_to_senders();
    }
    catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return ReturnStatus::failure;
    }

    m_obj_init_status = ReturnStatus::obj_init_success;
    return m_obj_init_status;
}

ReturnStatus IPMXSenderApp::run()
{
    if (m_obj_init_status != ReturnStatus::obj_init_success) {
        return m_obj_init_status;
    }

    ReturnStatus rc = run_stats_reader();
    if (rc == ReturnStatus::failure) {
        return ReturnStatus::failure;
    }

    try {
        run_threads(m_senders);
    }
    catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

bool IPMXSenderApp::device_has_ip(const rmx_device* device, const in_addr& addr)
{
    size_t ip_count = rmx_get_device_ip_count(device);
    for (size_t ip_index = 0; ip_index < ip_count; ip_index++) {
        const rmx_ip_addr *ip_addr = rmx_get_device_ip_address(device, ip_index);
        if (ip_addr == NULL) {
            std::cerr << "Error reading Rivermax device IP address" << std::endl;
            return false;
        }
        if ((ip_addr->family == AF_INET) && (ip_addr->addr.ipv4.s_addr == addr.s_addr)) {
            return true;
        }
    }
    return false;
}

ReturnStatus IPMXSenderApp::read_device_mac_address(const rmx_device* device, std::string& mac)
{
    const uint8_t *mac_addr = rmx_get_device_mac_address(device);
    if (mac_addr == NULL) {
        std::cerr << "Error reading Rivermax device MAC address" << std::endl;
        return ReturnStatus::failure;
    }
    std::stringstream s;
    s << std::hex << std::setw(2) << std::setfill('0');
    for (int octet = 0; octet < 6; octet++) {
        if (octet > 0) {
            s << "-";
        }
        s << static_cast<unsigned>(mac_addr[octet]);
    }
    mac = s.str();
    return ReturnStatus::success;
}

ReturnStatus IPMXSenderApp::read_local_mac_address(std::string& mac) const
{
    ReturnStatus rc = ReturnStatus::failure;
    rmx_device_list* device_list;
    size_t device_count = rmx_get_device_list(&device_list);
    if (device_count == 0) {
        std::cerr << "Rivermax device list is empty" << std::endl;
    }
    for (size_t device_index = 0; device_index < device_count; device_index++) {
        const rmx_device* device = rmx_get_device(device_list, device_index);
        if (device == NULL) {
            std::cerr << "Error reading Rivermax device list" << std::endl;
            break;
        }
        if (device_has_ip(device, m_local_address.sin_addr)) {
            rc = read_device_mac_address(device, mac);
            break;
        }
    }
    rmx_free_device_list(device_list);
    return rc;
}

ReturnStatus IPMXSenderApp::initialize_connection_parameters()
{
    ReturnStatus rc = RmaxBaseApp::initialize_connection_parameters();
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = init_device_iface(m_device_interface);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    return read_local_mac_address(m_app_settings->local_mac);
}

ReturnStatus IPMXSenderApp::init_device_iface(rmx_device_iface& device_iface)
{
    rmx_status status = rmx_retrieve_device_iface_ipv4(&device_iface, &m_local_address.sin_addr);
    if (status != RMX_OK) {
        char str[INET_ADDRSTRLEN];
        const char* s = inet_ntop(AF_INET, &(m_local_address.sin_addr), str, INET_ADDRSTRLEN);
        std::cerr << "Failed to get device: " << (s ? str : "unknown") << " with status: "
                  << status << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

class TimeContext
{
private:
    bool m_ref_clk_is_ptp;
    uint64_t m_nic_t0_ns;
    uint64_t m_wall_t0_ns;
    TimeContext() :
        m_ref_clk_is_ptp{false},
        m_nic_t0_ns{0},
        m_wall_t0_ns{0}
    {};
public:
    static TimeContext& get()
    {
        static TimeContext tc;
        return tc;
    }

    ReturnStatus init(bool ref_clk_is_ptp)
    {
        m_ref_clk_is_ptp = ref_clk_is_ptp;
        rmx_status status = rmx_get_time(RMX_TIME_PTP, &m_nic_t0_ns);
        if (status != RMX_OK) {
            return ReturnStatus::failure;
        }
        auto tai_time_now = (std::chrono::system_clock::now() + std::chrono::seconds{ LEAP_SECONDS }).time_since_epoch();
        m_wall_t0_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tai_time_now).count();
        return ReturnStatus::success;
    }

    uint64_t get_nic_time_ns()
    {
        uint64_t time_ns;
        if (rmx_get_time(RMX_TIME_PTP, &time_ns) != RMX_OK) {
            return 0;
        }
        return time_ns;
    }

    uint64_t get_wall_time_ns()
    {
        uint64_t nic_time_ns;
        if (rmx_get_time(RMX_TIME_PTP, &nic_time_ns) != RMX_OK) {
            return 0;
        }
        if (m_ref_clk_is_ptp) {
            return nic_time_ns;
        } else {
            return m_wall_t0_ns + (nic_time_ns - m_nic_t0_ns);
        }
    }
};

uint64_t get_nic_time_ns_cb(void* context)
{
    NOT_IN_USE(context);
    TimeContext& tc = TimeContext::get();
    return tc.get_nic_time_ns();
}

uint64_t get_wall_time_ns_cb(void* context)
{
    NOT_IN_USE(context);
    TimeContext& tc = TimeContext::get();
    return tc.get_wall_time_ns();
}

ReturnStatus IPMXSenderApp::set_rivermax_clock()
{
    std::cout << "Switching to PTP clock" << std::endl;
    ReturnStatus rc = set_rivermax_ptp_clock(&m_device_interface);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    TimeContext& tc = TimeContext::get();
    return tc.init(m_app_settings->ref_clk_is_ptp);
}

void IPMXSenderApp::initialize_send_flows()
{
    auto rc = initialize_media_settings(*m_app_settings, {{"IPMX", "", true}});
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to initialize media settings" << std::endl;
        return;
    }

    auto ip_octets = CLI::detail::split(m_app_settings->destination_ip, '.');
    auto ip_prefix = std::string(ip_octets[0] + "." + ip_octets[1] + "." + ip_octets[2] + ".");
    auto ip_last_octet = std::stoi(ip_octets[3]);
    m_stream_dst_addresses.reserve(m_app_settings->num_of_total_flows);

    for (size_t flow_index = 0; flow_index < m_app_settings->num_of_total_flows; flow_index++) {
        std::ostringstream ip;
        uint16_t port;
        ip << ip_prefix << (ip_last_octet + flow_index) % IP_OCTET_LEN;
        port = m_app_settings->destination_port;
        m_stream_dst_addresses.push_back(TwoTupleFlow(flow_index, ip.str(), port));
    }
}

void IPMXSenderApp::assign_streams_to_threads()
{
    m_streams_per_thread.reserve(m_app_settings->num_of_threads);
    for (int stream = 0; stream < m_app_settings->num_of_total_streams; stream++) {
        m_streams_per_thread[stream % m_app_settings->num_of_threads]++;
    }
}

void IPMXSenderApp::initialize_sender_threads()
{
    size_t streams_offset = 0;
    for (size_t sender_index = 0; sender_index < m_app_settings->num_of_threads; sender_index++) {
        int sender_cpu_core;
        if (sender_index < m_app_settings->app_threads_cores.size()) {
            sender_cpu_core = m_app_settings->app_threads_cores[sender_index];
        } else {
            std::cerr << "Warning: CPU affinity for Sender " << sender_index <<
                         " is not set!!!" << std::endl;
            sender_cpu_core = CPU_NONE;
        }
        auto src_address = TwoTupleFlow(
            sender_index,
            m_app_settings->local_ip,
            m_app_settings->source_port);
        auto flows = std::vector<TwoTupleFlow>(
            m_stream_dst_addresses.begin() + streams_offset,
            m_stream_dst_addresses.begin() + streams_offset + m_streams_per_thread[sender_index]);
        m_senders.push_back(std::make_unique<IPMXSenderIONode>(
            src_address,
            flows,
            m_app_settings,
            sender_index,
            sender_cpu_core,
            get_nic_time_ns_cb,
            get_wall_time_ns_cb));
        streams_offset += m_streams_per_thread[sender_index];
    }
}

size_t IPMXSenderApp::query_memory_size()
{
    size_t app_mem_len = 0;
    for (auto& sender : m_senders) {
        size_t tx_size;
        sender->query_memory_size(tx_size);
        app_mem_len += tx_size;
    }

    std::cout << "Application requires " << app_mem_len << " bytes of memory for Send Reports" << std::endl;
    return app_mem_len;
}

void* IPMXSenderApp::allocate_and_align_payload(size_t size)
{
    size = m_payload_allocator->align_length(size);
    return m_payload_allocator->allocate_aligned(size, m_payload_allocator->get_page_size());
}

ReturnStatus IPMXSenderApp::allocate_app_memory()
{
    size_t length = query_memory_size();
    rmx_mem_region mreg;

    memset(&mreg, 0, sizeof(mreg));
    mreg.addr = allocate_and_align_payload(length);
    mreg.length = length;
    mreg.mkey = 0;

    if (!mreg.addr) {
        std::cerr << "Failed to allocate application memory" << std::endl;
        return ReturnStatus::failure;
    }

    rmx_mem_reg_params mem_registry;
    rmx_init_mem_registry(&mem_registry, &m_device_interface);
    rmx_status status = rmx_register_memory(&mreg, &mem_registry);
    if (status != RMX_OK) {
        std::cerr << "Failed to register payload memory with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
    m_mem_region = mreg;

    std::cout << "Allocated " << mreg.length <<
        " bytes at address " << mreg.addr <<
        " with mkey: " << mreg.mkey << std::endl;

    return ReturnStatus::success;
}

void IPMXSenderApp::distribute_memory_to_senders()
{
    byte_t* pointer = reinterpret_cast<byte_t*>(m_mem_region.addr);
    rmx_mkey_id mkey = m_mem_region.mkey;
    for (auto& sender : m_senders) {
        size_t length = sender->initialize_memory(pointer, mkey);
        pointer += length;
    }
}
