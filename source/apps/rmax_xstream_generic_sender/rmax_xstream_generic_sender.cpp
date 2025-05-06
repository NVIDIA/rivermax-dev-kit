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

#include <cstring>

#include "rdk/apps/rmax_xstream_generic_sender/rmax_xstream_generic_sender.h"

using namespace rivermax::dev_kit::apps::rmax_xstream_generic_sender;

void GenericSenderSettings::init_default_values()
{
    AppSettings::init_default_values();
    app_memory_alloc = true;
    register_memory = true;
}

ReturnStatus GenericSenderSettingsValidator::validate(const std::shared_ptr<GenericSenderSettings>& settings) const
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
    if (settings->rate.bps > 0 && settings->rate.max_burst_in_packets == 0) {
        std::cerr << "Maximum allowed burst size per flow must be greater than 0 when Rate limit is set" << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

ReturnStatus GenericSenderCLISettingsBuilder::add_cli_options(std::shared_ptr<GenericSenderSettings>& settings)
{
    if (m_cli_parser_manager == nullptr) {
        std::cerr << "CLI parser manager is not initialized" << std::endl;
        return ReturnStatus::failure;
    }
    m_cli_parser_manager->add_option(CLIOptStr::LOCAL_IP);
    m_cli_parser_manager->add_option(CLIOptStr::DST_IP);
    m_cli_parser_manager->add_option(CLIOptStr::DST_PORT);
    m_cli_parser_manager->add_option(CLIOptStr::THREADS);
    m_cli_parser_manager->add_option(CLIOptStr::FLOWS);
    m_cli_parser_manager->add_option(CLIOptStr::STREAMS)->check(StreamToThreadsFlowsValidator(
        settings->num_of_threads, settings->num_of_total_flows));
    m_cli_parser_manager->add_option(CLIOptStr::VERBOSE);
    m_cli_parser_manager->add_option(CLIOptStr::CHUNKS);
    m_cli_parser_manager->add_option(CLIOptStr::PACKETS);
    m_cli_parser_manager->add_option(CLIOptStr::PAYLOAD_SIZE);
    m_cli_parser_manager->add_option(CLIOptStr::SLEEP_US);
    auto* rate_bps_opt = m_cli_parser_manager->add_option(CLIOptStr::RATE_BPS);
    auto* rate_burst_opt = m_cli_parser_manager->add_option(CLIOptStr::RATE_BURST);
    rate_bps_opt->needs(rate_burst_opt);
    rate_burst_opt->needs(rate_bps_opt);
    m_cli_parser_manager->add_option(CLIOptStr::INTERNAL_CORE);
    m_cli_parser_manager->add_option(CLIOptStr::APPLICATION_CORE);
    m_cli_parser_manager->add_option(CLIOptStr::WAIT_RETRY);
#ifdef CUDA_ENABLED
    m_cli_parser_manager->add_option(CLIOptStr::GPU_ID);
    m_cli_parser_manager->add_option(CLIOptStr::LOCK_GPU_CLOCKS);
#endif
    m_cli_parser_manager->add_option(CLIOptStr::ALLOCATOR_TYPE);
    return ReturnStatus::success;
}

GenericSenderApp::GenericSenderApp(std::shared_ptr<ISettingsBuilder<GenericSenderSettings>> settings_builder) :
    RmaxBaseApp(),
    m_settings_builder(std::move(settings_builder))
{
}

ReturnStatus GenericSenderApp::post_load_settings()
{
    if (m_app_settings->use_checksum_header) {
        m_app_settings->packet_app_header_size = sizeof(ChecksumHeader);
    }
    return ReturnStatus::success;
}

ReturnStatus GenericSenderApp::initialize_app_settings()
{
    if (m_settings_builder == nullptr) {
        std::cerr << "Settings builder is not initialized" << std::endl;
        return ReturnStatus::failure;
    }
    m_generic_sender_settings = std::make_shared<GenericSenderSettings>();
    ReturnStatus rc = m_settings_builder->build(m_generic_sender_settings);
    if (rc == ReturnStatus::success) {
        m_app_settings = m_generic_sender_settings;
        return ReturnStatus::success;
    }
    if (rc != ReturnStatus::success_cli_help) {
        std::cerr << "Failed to build settings" << std::endl;
    }
    m_obj_init_status = rc;
    return rc;
}

ReturnStatus GenericSenderApp::initialize()
{
    ReturnStatus rc  = RmaxBaseApp::initialize();

    if (rc != ReturnStatus::obj_init_success) {
        return m_obj_init_status;
    }

    try {
        distribute_work_for_threads();
        initialize_send_flows();
        initialize_sender_threads();
        rc = allocate_app_memory();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to allocate the memory required for the application" << std::endl;
            return rc;
        }
        const int mem_block_indx = 0;
        distribute_memory_for_senders(mem_block_indx);
    }
    catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return ReturnStatus::failure;
    }

    m_obj_init_status = ReturnStatus::obj_init_success;
    return m_obj_init_status;
}

ReturnStatus GenericSenderApp::run()
{
    if (m_obj_init_status != ReturnStatus::obj_init_success) {
        return m_obj_init_status;
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

ReturnStatus GenericSenderApp::cleanup_rivermax_resources()
{
    rmx_device_iface device_iface;
    rmx_status status = rmx_retrieve_device_iface_ipv4(&device_iface, &m_local_address.sin_addr);
    if (status != RMX_OK) {
        char str[INET_ADDRSTRLEN];
        const char* s = inet_ntop(AF_INET, &(m_local_address.sin_addr), str, INET_ADDRSTRLEN);
        std::cerr << "Failed to get device: " << (s ? str : "unknown") << " with status: " << status << std::endl;
        return ReturnStatus::failure;
    }

    for (auto& mreg : m_mem_regions) {
        status = rmx_deregister_memory(&mreg, &device_iface);
        if (status != RMX_OK) {
            std::cerr << "Failed to de-register application memory with status: "
                << status << std::endl;
            return ReturnStatus::failure;
        }
    }

    return ReturnStatus::success;
}

void GenericSenderApp::initialize_send_flows()
{
    auto ip_vec = CLI::detail::split(m_app_settings->destination_ip, '.');
    auto ip_prefix_str = std::string(ip_vec[0] + "." + ip_vec[1] + "." + ip_vec[2] + ".");
    auto ip_last_octet = std::stoi(ip_vec[3]);
    size_t flow_index = 0;
    std::ostringstream ip;
    uint16_t port;

    m_flows.reserve(m_app_settings->num_of_total_flows);
    while (flow_index != m_app_settings->num_of_total_flows) {
        ip << ip_prefix_str << (ip_last_octet + (flow_index / MAX_PORT)) % IP_OCTET_LEN;
        port = static_cast<uint16_t>(m_app_settings->destination_port + flow_index);
        m_flows.push_back(TwoTupleFlow(flow_index, ip.str(), port));
        ip.str("");
        flow_index++;
    }
}

ReturnStatus GenericSenderApp::allocate_app_memory()
{
    size_t length = get_memory_length();
    length = m_payload_allocator->align_length(length);
    rmx_mem_region mreg;

    memset(&mreg, 0, sizeof(mreg));
    mreg.addr = m_payload_allocator->allocate(length);
    mreg.length = length;
    mreg.mkey = 0;

    if (!mreg.addr) {
        std::cerr << "Failed to allocate application memory" << std::endl;
        return ReturnStatus::failure;
    }

    rmx_device_iface device_iface;
    rmx_status status = rmx_retrieve_device_iface_ipv4(&device_iface, &m_local_address.sin_addr);
    if (status != RMX_OK) {
        char str[INET_ADDRSTRLEN];
        const char* s = inet_ntop(AF_INET, &(m_local_address.sin_addr), str, INET_ADDRSTRLEN);
        std::cerr << "Failed to get device: " << (s ? str : "unknown") << " with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
    rmx_mem_reg_params mem_registry;
    rmx_init_mem_registry(&mem_registry, &device_iface);
    status = rmx_register_memory(&mreg, &mem_registry);
    if (status != RMX_OK) {
        std::cerr << "Failed to register payload memory with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
    m_mem_regions.push_back(mreg);

    std::cout << "Allocated " << mreg.length <<
        " bytes at address " << mreg.addr <<
        " with mkey: " << mreg.mkey << std::endl;

    return ReturnStatus::success;
}

size_t GenericSenderApp::get_memory_length()
{
    size_t app_mem_len = 0;
    for (auto& sender : m_senders) {
        for (auto& stream : sender->get_streams()) {
            app_mem_len += stream->get_memory_length();
        }
    }

    std::cout << "Application requires " << app_mem_len << " bytes of memory" << std::endl;
    return app_mem_len;
}

void GenericSenderApp::distribute_work_for_threads()
{
    m_streams_per_thread.reserve(m_app_settings->num_of_threads);
    m_flows_per_stream.reserve(m_app_settings->num_of_total_streams);
    m_flows_per_thread.reserve(m_app_settings->num_of_threads);
    for (int stream = 0; stream < m_app_settings->num_of_total_streams; stream++) {
        m_streams_per_thread[stream % m_app_settings->num_of_threads]++;
    }
    for (size_t flow = 0; flow < m_app_settings->num_of_total_flows; flow++) {
        m_flows_per_stream[flow % m_app_settings->num_of_total_streams]++;
        m_flows_per_thread[flow % m_app_settings->num_of_threads]++;
    }
}

void GenericSenderApp::initialize_sender_threads()
{
    size_t flows_offset = 0;

    for (size_t sndr_indx = 0; sndr_indx < m_app_settings->num_of_threads; sndr_indx++) {
        int sender_cpu_core = m_app_settings->app_threads_cores[sndr_indx % m_app_settings->app_threads_cores.size()];
        auto flows = std::vector<TwoTupleFlow>(
            m_flows.begin() + flows_offset,
            m_flows.begin() + flows_offset + m_flows_per_thread[sndr_indx]);
        m_senders.push_back(std::unique_ptr<GenericSenderIONode>(new GenericSenderIONode(
            m_app_settings,
            sndr_indx,
            m_streams_per_thread[sndr_indx],
            sender_cpu_core,
            m_payload_allocator->get_memory_utils())));
        m_senders[sndr_indx]->initialize_send_flows(flows);
        m_senders[sndr_indx]->initialize_streams(flows_offset);
    }
}

void GenericSenderApp::distribute_memory_for_senders(const int mem_block_index)
{
    auto& app_mem = m_mem_regions[mem_block_index];
    byte_t* pointer = nullptr;
    rmx_mkey_id mkey = 0;
    size_t length = 0;
    size_t offset = 0;

    for (auto& sender : m_senders) {
        pointer = reinterpret_cast<byte_t*>(app_mem.addr) + offset;
        mkey = app_mem.mkey;
        length = sender->initialize_memory(pointer, mkey);
        offset += length;
    }
}
