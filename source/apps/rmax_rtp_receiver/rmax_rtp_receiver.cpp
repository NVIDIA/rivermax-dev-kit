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

#include "rdk/apps/rmax_rtp_receiver/rmax_rtp_receiver.h"

using namespace rivermax::dev_kit::apps::rmax_rtp_receiver;

void RTPReceiverSettings::init_default_values()
{
    AppSettings::init_default_values();
    app_memory_alloc = true;
    num_of_packets_in_chunk = DEFAULT_NUM_OF_PACKETS_IN_CHUNK;
    is_extended_sequence_number = false;
}

ReturnStatus RTPReceiverSettingsValidator::validate(const std::shared_ptr<RTPReceiverSettings>& settings) const
{
#if defined(CUDA_ENABLED) && !defined(TEGRA_ENABLED)
    if (settings->gpu_id != INVALID_GPU_ID && settings->packet_app_header_size == 0) {
        std::cerr << "GPU Direct is supported only in header-data split mode\n"
                << "Please specify application header size." << std::endl;
        return ReturnStatus::failure;
    }
#endif

    ReturnStatus rc = ValidatorUtils::validate_ip4_address(settings->source_ip);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_ip4_address(settings->local_ip);
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

    return ReturnStatus::success;
}

ReturnStatus RTPReceiverCLISettingsBuilder::add_cli_options(std::shared_ptr<RTPReceiverSettings>& settings)
{
    if (m_cli_parser_manager == nullptr) {
        std::cerr << "CLI parser manager is not initialized" << std::endl;
        return ReturnStatus::failure;
    }
    m_cli_parser_manager->add_option(CLIOptStr::SRC_IP);
    m_cli_parser_manager->add_option(CLIOptStr::DST_IP);
    m_cli_parser_manager->add_option(CLIOptStr::LOCAL_IP);
    m_cli_parser_manager->add_option(CLIOptStr::DST_PORT);
    m_cli_parser_manager->add_option(CLIOptStr::THREADS);
    m_cli_parser_manager->add_option(CLIOptStr::STREAMS);
    m_cli_parser_manager->add_option(CLIOptStr::PACKETS);
    m_cli_parser_manager->add_option(CLIOptStr::PAYLOAD_SIZE);
    m_cli_parser_manager->add_option(CLIOptStr::APP_HDR_SIZE);
    m_cli_parser_manager->add_option(CLIOptStr::INTERNAL_CORE);
    m_cli_parser_manager->add_option(CLIOptStr::APPLICATION_CORE);
    m_cli_parser_manager->add_option(CLIOptStr::SLEEP_US);
#ifdef CUDA_ENABLED
    m_cli_parser_manager->add_option(CLIOptStr::GPU_ID);
    m_cli_parser_manager->add_option(CLIOptStr::LOCK_GPU_CLOCKS);
#endif
    m_cli_parser_manager->add_option(CLIOptStr::ALLOCATOR_TYPE);
    m_cli_parser_manager->add_option(CLIOptStr::REGISTER_MEMORY);
    m_cli_parser_manager->add_option(CLIOptStr::VERBOSE);
    m_cli_parser_manager->add_option(CLIOptStr::STATS_REPORT_INTERVAL);

    CLI::App_p parser = m_cli_parser_manager->get_parser();
    parser->add_flag("-X,--ext-seq-num", settings->is_extended_sequence_number,
        "Parse extended sequence number from RTP payload");
    return ReturnStatus::success;
}

RTPReceiverApp::RTPReceiverApp(std::shared_ptr<ISettingsBuilder<RTPReceiverSettings>> settings_builder) :
    RmaxReceiverBaseApp(),
    m_settings_builder(std::move(settings_builder))
{
    m_num_paths_per_stream = 1;
}

ReturnStatus RTPReceiverApp::initialize_connection_parameters()
{
    m_device_interfaces.resize(1);
    in_addr device_address;
    if (inet_pton(AF_INET, m_app_settings->local_ip.c_str(), &device_address) != 1) {
        std::cerr << "Failed to parse address of device " << m_app_settings->local_ip << std::endl;
        return ReturnStatus::failure;
    }
    rmx_status status = rmx_retrieve_device_iface_ipv4(&m_device_interfaces[0], &device_address);
    if (status != RMX_OK) {
        std::cerr << "Failed to get device: " << m_app_settings->local_ip << " with status: " << status << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}
ReturnStatus RTPReceiverApp::initialize_app_settings()
{
    if (m_settings_builder == nullptr) {
        std::cerr << "Settings builder is not initialized" << std::endl;
        return ReturnStatus::failure;
    }
    m_rtp_receiver_settings = std::make_shared<RTPReceiverSettings>();
    ReturnStatus rc = m_settings_builder->build(m_rtp_receiver_settings);
    if (rc == ReturnStatus::success) {
        m_app_settings = m_rtp_receiver_settings;
        return ReturnStatus::success;
    }
    if (rc != ReturnStatus::success_cli_help) {
        std::cerr << "Failed to build settings" << std::endl;
    }
    m_obj_init_status = rc;
    return rc;
}

void RTPReceiverApp::run_receiver_threads()
{
    run_threads(m_receivers);
}

void RTPReceiverApp::configure_network_flows()
{
    std::string ip_prefix_str;
    int ip_last_octet;
    uint16_t src_port = 0;

    auto ip_vec = CLI::detail::split(m_app_settings->destination_ip, '.');
    ip_prefix_str = std::string(ip_vec[0] + "." + ip_vec[1] + "." + ip_vec[2] + ".");
    ip_last_octet = std::stoi(ip_vec[3]);

    m_flows.reserve(m_app_settings->num_of_total_streams);
    size_t id = 0;
    for (size_t flow_index = 0; flow_index < m_app_settings->num_of_total_streams; ++flow_index) {
        std::ostringstream ip;
        ip << ip_prefix_str << (ip_last_octet + flow_index) % IP_OCTET_LEN;
        ReceiveFlow flow(id++, m_app_settings->source_ip, src_port, ip.str(), m_app_settings->destination_port);
        m_flows.push_back(flow);
    }
}

void RTPReceiverApp::initialize_receive_io_nodes()
{
    size_t streams_offset = 0;
    for (size_t rx_idx = 0; rx_idx < m_app_settings->num_of_threads; rx_idx++) {
        int recv_cpu_core = m_app_settings->app_threads_cores[rx_idx % m_app_settings->app_threads_cores.size()];

        auto flows = std::vector<ReceiveFlow>(
            m_flows.begin() + streams_offset,
            m_flows.begin() + streams_offset + m_streams_per_thread[rx_idx]);
        m_receivers.push_back(std::unique_ptr<ReceiverIONodeBase>(new RTPReceiverIONode(
            *m_app_settings,
            m_rtp_receiver_settings->is_extended_sequence_number,
            { m_app_settings->local_ip },
            rx_idx,
            recv_cpu_core,
            *m_memory_utils)));
        static_cast<RTPReceiverIONode*>(m_receivers[rx_idx].get())->initialize_streams(streams_offset, flows);
        m_receivers[rx_idx]->set_statistics_report_interval(m_app_settings->stats_report_interval_ms);
        streams_offset += m_streams_per_thread[rx_idx];
    }
}
