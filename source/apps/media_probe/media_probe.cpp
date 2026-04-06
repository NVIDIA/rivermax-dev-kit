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

#include "rdk/apps/media_probe/media_probe.h"
#include "rdk/apps/media_probe/stream_monitor.h"
#include <unordered_set>

using namespace rdk::apps;

void MediaProbeSettings::init_default_values()
{
    AppSettings::init_default_values();
    app_memory_alloc = true;
    num_of_packets_in_chunk = DEFAULT_NUM_OF_PACKETS_IN_CHUNK;
    register_memory = true;
}

ReturnStatus MediaProbeSettingsValidator::validate(const MediaProbeSettings& settings) const
{
    ReturnStatus rc = ValidatorUtils::validate_ip4_address(settings.source_ip);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_ip4_address(settings.local_ip);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_ip4_address(settings.destination_ip);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_ip4_port(settings.destination_port);
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

    return ReturnStatus::success;
}

ReturnStatus MediaProbeCLISettingsBuilder::add_cli_options(MediaProbeSettings& settings)
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
    m_cli_parser_manager->add_option(CLIOptStr::INTERNAL_CORE);
    m_cli_parser_manager->add_option(CLIOptStr::APPLICATION_CORE);
    m_cli_parser_manager->add_option(CLIOptStr::SLEEP_US);
    m_cli_parser_manager->add_option(CLIOptStr::ALLOCATOR_TYPE);
    m_cli_parser_manager->add_option(CLIOptStr::VERBOSE);
    m_cli_parser_manager->add_option(CLIOptStr::STATS_REPORT_INTERVAL);

    CLI::App_p parser = m_cli_parser_manager->get_parser();
    auto video_cli_flag = parser->add_flag("--video", settings.is_video_enabled,
        "Parse video stream");
    auto alpha_cli_flag = parser->add_flag("--alpha", settings.is_alpha_enabled,
        "Parse alpha stream");
    alpha_cli_flag->needs(video_cli_flag);

    return ReturnStatus::success;
}

MediaProbeApp::MediaProbeApp(std::unique_ptr<ISettingsBuilder<MediaProbeSettings>> settings_builder) :
    ReceiverBaseApp(),
    m_settings_builder(std::move(settings_builder))
{
    m_num_paths_per_stream = 1;
}

ReturnStatus MediaProbeApp::set_rivermax_clock()
{
    std::cout << "Switching to PTP clock" << std::endl;
    return set_rivermax_ptp_clock(&m_device_interfaces[0]);
}

ReturnStatus MediaProbeApp::initialize_app_settings()
{
    if (m_settings_builder == nullptr) {
        std::cerr << "Settings builder is not initialized" << std::endl;
        return ReturnStatus::failure;
    }
    m_media_probe_settings = std::make_shared<MediaProbeSettings>();
    ReturnStatus rc = m_settings_builder->build(*m_media_probe_settings);
    if (rc == ReturnStatus::success) {
        m_app_settings = m_media_probe_settings;
        return ReturnStatus::success;
    }
    if (rc != ReturnStatus::success_cli_help) {
        std::cerr << "Failed to build settings" << std::endl;
    }
    m_obj_init_status = rc;
    return rc;
}

void MediaProbeApp::configure_network_flows()
{
    std::string ip_prefix_str;
    int ip_last_octet;
    uint16_t src_port = 0;

    auto ip_vec = CLI::detail::split(m_app_settings->destination_ip, '.');
    ip_prefix_str = std::string(ip_vec[0] + "." + ip_vec[1] + "." + ip_vec[2] + ".");
    ip_last_octet = std::stoi(ip_vec[3]);

    auto& probe_app_settings = static_cast<MediaProbeSettings&>(*m_app_settings);
    size_t flow_index = 0;

    if (probe_app_settings.is_video_enabled) {
        m_video_flows.reserve(m_app_settings->num_of_total_streams);
        for (size_t video_flow_idx = 0; video_flow_idx < m_app_settings->num_of_total_streams; ++video_flow_idx) {
            std::ostringstream ip;
            ip << ip_prefix_str << (ip_last_octet + flow_index) % IP_OCTET_LEN;
            ReceiveFlow flow(flow_index++, m_app_settings->source_ip, src_port, ip.str(), m_app_settings->destination_port);
            m_video_flows.push_back(flow);
        }
    }
    if (probe_app_settings.is_alpha_enabled) {
        m_alpha_flows.reserve(m_app_settings->num_of_total_streams);
        for (size_t alpha_flow_idx = 0; alpha_flow_idx < m_app_settings->num_of_total_streams; ++alpha_flow_idx) {
            std::ostringstream ip;
            ip << ip_prefix_str << (ip_last_octet + flow_index) % IP_OCTET_LEN;
            ReceiveFlow flow(flow_index++, m_app_settings->source_ip, src_port, ip.str(), m_app_settings->destination_port);
            m_alpha_flows.push_back(flow);
        }
    }
}

void MediaProbeApp::initialize_media_probe_node_streams(RTPReceiverIONode& node, size_t start_media_index,
    const std::vector<ReceiveFlow>& flows, MediaComponentId component_index)
{
    std::vector<std::unique_ptr<IReceiveStream>> streams;
    streams.reserve(flows.size());
    size_t id = start_media_index;
    for (size_t stream_index = 0; stream_index < flows.size(); stream_index++) {
        ReceiveStreamSettings stream_settings(TwoTupleFlow(id++, m_app_settings->local_ip, 0),
            RMX_INPUT_APP_PROTOCOL_PACKET,
            RMX_INPUT_TIMESTAMP_SYNCED,
            {RMX_INPUT_STREAM_CREATE_INFO_PER_PACKET},
            m_app_settings->num_of_packets_in_chunk,
            m_app_settings->packet_payload_size,
            m_app_settings->packet_app_header_size,
            0, RECEIVE_CHUNK_SIZE_LIMIT);
        auto stream = std::make_unique<AppRTPReceiveStream>(stream_settings,
            m_is_extended_sequence_number,
            false, false);
        streams.push_back(std::move(stream));
    }
    node.assign_streams(start_media_index, flows, streams);
    size_t media_index = start_media_index;
    for (size_t stream_index = 0; stream_index < flows.size(); stream_index++) {
        auto stream_monitor = std::make_unique<StreamMonitor>(*m_media_probe_settings, flows[stream_index], media_index, component_index);
        m_media_monitors[media_index]->add_stream_monitor(*stream_monitor);
        ReturnStatus rc = node.set_receive_data_consumer(stream_index, std::make_unique<StreamMonitorAdapter>(*stream_monitor));
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to set data consumer for stream " << stream_index << std::endl;
        }
        m_stream_monitors.push_back(std::move(stream_monitor));
        media_index++;
    }
    node.set_statistics_report_interval(m_app_settings->stats_report_interval_ms);
}

void MediaProbeApp::initialize_component_receivers(size_t& receiver_index, MediaComponentId component_index, const std::vector<ReceiveFlow>& flows)
{
    size_t start_media_index = 0;
    for (size_t rx_index = 0; rx_index < m_app_settings->num_of_threads; rx_index++) {
        int recv_cpu_core = m_app_settings->app_threads_cores[receiver_index % m_app_settings->app_threads_cores.size()];

        auto component_flows = std::vector<ReceiveFlow>(
            flows.begin() + start_media_index,
            flows.begin() + start_media_index + m_streams_per_thread[rx_index]);
        auto receiver = std::make_unique<RTPReceiverIONode>(
            *m_app_settings,
            true,
            std::vector<std::string>{ m_app_settings->local_ip },
            receiver_index,
            recv_cpu_core,
            *m_memory_utils);
        initialize_media_probe_node_streams(*receiver, start_media_index, component_flows, component_index);
        m_receivers.push_back(std::move(receiver));
        start_media_index += m_streams_per_thread[rx_index];
        receiver_index++;
    }
}

void MediaProbeApp::initialize_receive_io_nodes()
{
    size_t receiver_index = 0;

    m_media_monitors.reserve(m_app_settings->num_of_total_streams);
    for (size_t media_monitor_index = 0; media_monitor_index < m_app_settings->num_of_total_streams; ++media_monitor_index) {
        m_media_monitors.push_back(std::make_unique<MediaMonitor>(media_monitor_index));
    }

    if (m_media_probe_settings->is_video_enabled) {
        initialize_component_receivers(receiver_index, MediaComponentId::Video, m_video_flows);
    }

    if (m_media_probe_settings->is_alpha_enabled) {
        initialize_component_receivers(receiver_index, MediaComponentId::Alpha, m_alpha_flows);
    }
}

void MediaProbeApp::run_receiver_threads()
{
    for (auto& io_node : m_receivers) {
        m_threads.push_back(std::thread(std::ref(*io_node)));
    }

    while (likely(SignalHandler::get_received_signal() < 0)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(m_app_settings->stats_report_interval_ms));
        for (auto& media_monitor : m_media_monitors) {
            media_monitor->print_and_reset_stats(std::cout);
        }
        std::cout << "--------------------------------" << std::endl;
    }

    for (auto& thread : m_threads) {
        thread.join();
    }
}
