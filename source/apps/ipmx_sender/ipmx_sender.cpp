/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "rdk/facade.h"
#include "rdk/apps/ipmx_sender/ipmx_sender.h"
#include "rdk/services/sdp/sdp_common_descriptions.h"
#include "rdk/services/utils/clock.h"
#include "rdk/services/utils/defs.h"

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::apps::ipmx_sender;
using namespace rivermax::dev_kit::apps::ipmx_sender;

void IPMXSenderSettings::init_default_values()
{
    AppSettings::init_default_values();
    media.frames_fields_in_mem_block = DEFAULT_MEMORY_BUFFER_SIZE_IN_FRAMES;
    ref_clk_is_ptp = false;
    app_memory_alloc = true;
    register_memory = true;
    enabled_smpte_standards.insert(SMPTEStandard::ST_2110_20);
}

ReturnStatus IPMXSenderSettingsValidator::validate(const IPMXSenderSettings& settings) const
{
    if (settings.local_ips.empty() && settings.local_ip.empty()) {
        std::cerr << "At least one local IP must be specified" << std::endl;
        return ReturnStatus::failure;
    }
    if (!settings.local_ips.empty() && !settings.local_ip.empty()) {
        std::cerr << "Cannot set both a single local IP and a local IP list" << std::endl;
        return ReturnStatus::failure;
    }
    if (settings.local_ips.size() > rivermax::dev_kit::RivermaxDevKitFacade::get_max_redundant_streams()) {
        std::cerr << "Up to " << rivermax::dev_kit::RivermaxDevKitFacade::get_max_redundant_streams()
                  << " local IP addresses are supported" << std::endl;
        return ReturnStatus::failure;
    }
    if (settings.destination_ips.size() != settings.local_ips.size()) {
        std::cerr << "Must be the same number of destination multicast IPs as number of local IPs" << std::endl;
        return ReturnStatus::failure;
    }
    if (settings.destination_ports.size() != settings.local_ips.size()) {
        std::cerr << "Must be the same number of destination ports as number of local IPs" << std::endl;
        return ReturnStatus::failure;
    }
    ReturnStatus rc = ValidatorUtils::validate_ip4_address(settings.local_ips);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_ip4_address(settings.destination_ips);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_ip4_port(settings.destination_ports);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_core(settings.internal_thread_core);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_core(settings.app_threads_cores);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_core(settings.statistics_reader_core);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    if (settings.register_memory && !settings.app_memory_alloc) {
        std::cerr << "Register memory option is supported only with application memory allocation" << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

ReturnStatus IPMXSenderCLISettingsBuilder::add_cli_options(IPMXSenderSettings& settings)
{
    if (m_cli_parser_manager == nullptr) {
        std::cerr << "CLI parser manager is not initialized" << std::endl;
        return ReturnStatus::failure;
    }
    auto cli_option_ip = m_cli_parser_manager->add_option(CLIOptStr::LOCAL_IP);
    auto cli_option_ips = m_cli_parser_manager->add_option(CLIOptStr::LOCAL_IPS);
    cli_option_ip->excludes(cli_option_ips);
    cli_option_ips->excludes(cli_option_ip);
    m_cli_parser_manager->add_option(CLIOptStr::DST_IP)->needs(cli_option_ip);
    m_cli_parser_manager->add_option(CLIOptStr::DST_IPS)->needs(cli_option_ips);
    m_cli_parser_manager->add_option(CLIOptStr::DST_PORT)->needs(cli_option_ip);
    m_cli_parser_manager->add_option(CLIOptStr::DST_PORTS)->needs(cli_option_ips);
    m_cli_parser_manager->add_option(CLIOptStr::THREADS);
    m_cli_parser_manager->add_option(CLIOptStr::STREAMS)->check(
        StreamToThreadsValidator(settings.num_of_threads));
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
    m_cli_parser_manager->add_option(CLIOptStr::VIDEO_SCAN_TYPE)
        ->group(CLIGroupStr::VIDEO_FORMAT_OPTIONS);
    // TODO: move PTP flag to App
    parser->add_flag("--ptp", settings.ref_clk_is_ptp,
                     "Use NIC RTC as a PTP-synchronized Common Reference clock");
    auto stats_enabled = m_cli_parser_manager->add_option(CLIOptStr::ENABLE_STATS_READER);
    auto stats_core = m_cli_parser_manager->add_option(CLIOptStr::STATS_CORE)->needs(stats_enabled);
    m_cli_parser_manager->add_option(CLIOptStr::STATS_SESSION_ID)->needs(stats_enabled);
    return ReturnStatus::success;
}

IPMXSenderApp::IPMXSenderApp(std::unique_ptr<ISettingsBuilder<IPMXSenderSettings>> settings_builder) :
    BaseApp(),
    m_settings_builder(std::move(settings_builder)),
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
    ReturnStatus rc = m_settings_builder->build(*m_ipmx_sender_settings);
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
    ReturnStatus rc  = BaseApp::initialize();
    if (rc != ReturnStatus::success) {
        return rc;
    }

    try {
        rc = configure_smpte_standards_processing();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to configure SMPTE standards" << std::endl;
            return rc;
        }
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
    return ReturnStatus::success;
}

ReturnStatus IPMXSenderApp::post_load_settings()
{
    if (m_app_settings->local_ips.empty() && !m_app_settings->local_ip.empty()) {
        m_app_settings->local_ips.push_back(m_app_settings->local_ip);
    }
    if (m_app_settings->destination_ips.empty() && !m_app_settings->destination_ip.empty()) {
        m_app_settings->destination_ips.push_back(m_app_settings->destination_ip);
    }
    if (m_app_settings->destination_ports.empty()) {
        m_app_settings->destination_ports.push_back(m_app_settings->destination_port);
    }
    return ReturnStatus::success;
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

ReturnStatus IPMXSenderApp::read_local_mac_address(const sockaddr_in& local_address, std::string& mac) const
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
        if (device_has_ip(device, local_address.sin_addr)) {
            rc = read_device_mac_address(device, mac);
            break;
        }
    }
    rmx_free_device_list(device_list);
    return rc;
}

ReturnStatus IPMXSenderApp::read_local_mac_addresses()
{
    for (const auto& local_address : m_local_addresses) {
        std::string local_mac;
        ReturnStatus rc = read_local_mac_address(local_address, local_mac);
        if (rc != ReturnStatus::success) {
            return rc;
        }
        m_ipmx_sender_settings->local_macs.push_back(local_mac);
    }
    return ReturnStatus::success;
}

ReturnStatus IPMXSenderApp::initialize_connection_parameters()
{
    ReturnStatus rc = BaseApp::initialize_connection_parameters();
    if (rc != ReturnStatus::success) {
        return rc;
    }
    return read_local_mac_addresses();
}

ReturnStatus IPMXSenderApp::set_rivermax_clock()
{
    std::cout << "Switching to PTP clock" << std::endl;
    return set_rivermax_ptp_clock(&m_device_interfaces[0]);
}

void IPMXSenderApp::initialize_send_flows()
{
    constexpr bool dest_port_iteration = false;
    auto ip_vec = CLI::detail::split(m_app_settings->destination_ip, '.');
    auto ip_prefix_str = std::string(ip_vec[0] + "." + ip_vec[1] + "." + ip_vec[2] + ".");
    auto ip_last_octet = std::stoi(ip_vec[3]);
    size_t flow_index = 0;
    std::ostringstream ip;
    uint16_t port;

    size_t total_num_of_flows = 0;
    for (const auto& node : m_ipmx_sender_settings->smpte_standard_to_nodes) {
        total_num_of_flows += node.second;
    }
    m_stream_dst_addresses.reserve(total_num_of_flows);

    for (const auto& node : m_ipmx_sender_settings->smpte_standard_to_nodes) {
        auto& smpte_standard_config = node.first;
        auto& num_of_streams = node.second;
        for (size_t i = 0; i < num_of_streams; i++) {
            if (dest_port_iteration) {
                ip << m_app_settings->destination_ip;
                port = m_app_settings->destination_port + static_cast<uint16_t>(flow_index);
            } else {
                ip << ip_prefix_str << (ip_last_octet + flow_index) % IP_OCTET_LEN;
                port = m_app_settings->destination_port;
            }
            m_stream_dst_addresses.push_back(TwoTupleFlow(flow_index, ip.str(), port));
            ip.str("");
            flow_index++;
        }
    }
}

ReturnStatus IPMXSenderApp::configure_video_settings()
{
    static const std::vector<FormatSpecificParameter> extra_ipmx_parameters = {{"IPMX", "", true}};

    size_t num_of_video_threads = std::min<size_t>(m_app_settings->num_of_threads, m_app_settings->num_of_total_streams);
    if (num_of_video_threads < m_app_settings->num_of_threads) {
        std::cout << "The number of media sender threads is limited to the number of streams ("
            << num_of_video_threads << ")" << std::endl;
        m_app_settings->num_of_threads = num_of_video_threads;
    }
    size_t min_number_streams_per_thread = m_app_settings->num_of_total_streams / num_of_video_threads;

    m_ipmx_sender_settings->smpte_standard_to_nodes.clear();

    auto video_settings = std::make_unique<SMPTE_2110_20_MediaSettings>(*m_app_settings);
    auto rc = video_settings->create_default_calculator(extra_ipmx_parameters);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to create default calculator for 2110-20" << std::endl;
        return rc;
    }

    video_settings->ref_clk_is_ptp = m_app_settings->ref_clk_is_ptp;
    if (m_app_settings->ref_clk_is_ptp) {
        video_settings->refclk_id = "";
    } else {
        video_settings->refclk_id = m_ipmx_sender_settings->local_macs[0]; 
    }

    for (size_t idx = 0; idx < num_of_video_threads; idx++) {
        size_t num_of_streams_in_cur_thread;
        if (min_number_streams_per_thread * num_of_video_threads + idx < m_app_settings->num_of_total_streams) {
            num_of_streams_in_cur_thread = min_number_streams_per_thread + 1;
        } else {
            num_of_streams_in_cur_thread = min_number_streams_per_thread;
        }
        m_ipmx_sender_settings->smpte_standard_to_nodes.emplace_back(*video_settings, num_of_streams_in_cur_thread);
    }
    m_ipmx_sender_settings->smpte_standard_configs.push_back(std::move(video_settings));
    return ReturnStatus::success;
}

ReturnStatus IPMXSenderApp::configure_smpte_standards_processing()
{
    if (m_ipmx_sender_settings->enabled_smpte_standards.count(SMPTEStandard::ST_2110_20)) {
        return configure_video_settings();
    }
    std::cerr << "No supported SMPTE standards to configure" << std::endl;
    return ReturnStatus::failure;
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
            m_app_settings->local_ips[0],
            m_app_settings->source_port);
        auto flows = std::vector<TwoTupleFlow>(
            m_stream_dst_addresses.begin() + streams_offset,
            m_stream_dst_addresses.begin() + streams_offset + m_streams_per_thread[sender_index]);
        m_senders.push_back(std::make_unique<IPMXSenderIONode>(
            src_address,
            flows,
            m_app_settings,
            m_ipmx_sender_settings->smpte_standard_to_nodes[sender_index].first,
            sender_index,
            sender_cpu_core));
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
    rmx_init_mem_registry(&mem_registry, &m_device_interfaces[0]);
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
