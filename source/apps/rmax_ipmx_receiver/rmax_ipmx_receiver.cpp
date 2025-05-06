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

#include <memory>
#include <utility>
#include <vector>

#include <rivermax_api.h>

#include "rdk/apps/rmax_ipmx_receiver/rmax_ipmx_receiver.h"
#include "rdk/services/utils/defs.h"
#include "rdk/services/utils/clock.h"
#include "rdk/services/error_handling/return_status.h"

using namespace rivermax::dev_kit::apps::rmax_ipmx_receiver;

ReturnStatus RTCPChunkConsumer::consume_chunk(const ReceiveChunk& chunk,
    const IReceiveStream& stream, size_t& consumed_packets)
{
    size_t stride_size = stream.get_payload_stride_size();
    const byte_t* packet_ptr = reinterpret_cast<const byte_t*>(chunk.get_payload_ptr());
    for (uint32_t stride_index = 0; stride_index < chunk.get_length(); ++stride_index) {
        const ReceivePacketInfo& info = chunk.get_packet_info(stride_index);
        size_t len = info.get_packet_sub_block_size(0);
        uint32_t flow_tag = info.get_packet_flow_tag();
        assert(flow_tag < m_trackers.size());
        m_trackers[flow_tag]->consume_rtcp_packet(packet_ptr, info);
    }
    consumed_packets = chunk.get_length();
    return ReturnStatus::success;
}

void IPMXReceiverSettings::init_default_values()
{
    AppSettings::init_default_values();
    app_memory_alloc = true;
    num_of_packets_in_chunk = NUM_OF_PACKETS_IN_CHUNK_DEFAULT;
    is_extended_sequence_number = false;
    rtcp_thread_core = INVALID_CORE_NUMBER;
}

ReturnStatus IPMXReceiverSettingsValidator::validate(const std::shared_ptr<IPMXReceiverSettings>& settings) const
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
    rc = ValidatorUtils::validate_core(settings->rtcp_thread_core);
    if (rc != ReturnStatus::success) {
        return rc;
    }

    return ReturnStatus::success;
}

ReturnStatus IPMXReceiverCLISettingsBuilder::add_cli_options(std::shared_ptr<IPMXReceiverSettings>& settings)
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
    parser->add_option("-r,--rtcp-core", settings->rtcp_thread_core,
        "RTCP receiver thread core");
    return ReturnStatus::success;
}

IPMXReceiverApp::IPMXReceiverApp(std::shared_ptr<ISettingsBuilder<IPMXReceiverSettings>> settings_builder) :
    RmaxReceiverBaseApp(),
    m_settings_builder(std::move(settings_builder))
{
    m_num_paths_per_stream = 1;
}

ReturnStatus IPMXReceiverApp::set_rivermax_clock()
{
    std::cout << "Switching to using NIC realtime clock" << std::endl;
    return set_rivermax_ptp_clock(&m_device_interfaces[0]);
}

ReturnStatus IPMXReceiverApp::initialize_connection_parameters()
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

ReturnStatus IPMXReceiverApp::initialize_app_settings()
{
    if (m_settings_builder == nullptr) {
        std::cerr << "Settings builder is not initialized" << std::endl;
        return ReturnStatus::failure;
    }
    m_ipmx_receiver_settings = std::make_shared<IPMXReceiverSettings>();
    ReturnStatus rc = m_settings_builder->build(m_ipmx_receiver_settings);
    if (rc == ReturnStatus::success) {
        m_app_settings = m_ipmx_receiver_settings;
        initialize_rtcp_ionode_settings();
        return ReturnStatus::success;
    }
    if (rc != ReturnStatus::success_cli_help) {
        std::cerr << "Failed to build settings" << std::endl;
    }
    m_obj_init_status = rc;
    return rc;
}

void IPMXReceiverApp::initialize_rtcp_ionode_settings()
{
    m_rtcp_receiver_settings = *m_app_settings;
    m_rtcp_receiver_settings.destination_port += 1;
    m_rtcp_receiver_settings.header_data_split = false;
    m_rtcp_receiver_settings.packet_app_header_size = 0;
    m_rtcp_receiver_settings.packet_payload_size = PACKET_PAYLOAD_SIZE_DEFAULT;
    m_rtcp_receiver_settings.num_of_chunks_in_mem_block = 2 * m_app_settings->num_of_total_streams;
    m_rtcp_receiver_settings.num_of_packets_in_chunk = 1;
    m_rtcp_receiver_settings.num_of_packets_in_chunk_specified = 1;
    m_rtcp_receiver_settings.num_of_packets_in_mem_block =
        m_rtcp_receiver_settings.num_of_chunks_in_mem_block *
        m_rtcp_receiver_settings.num_of_packets_in_chunk;
    m_rtcp_receiver_settings.allocator_type = AllocatorTypeUI::Malloc;
}

void IPMXReceiverApp::run_receiver_threads()
{
    for (auto& io_node : m_receivers) {
        m_threads.push_back(std::thread(std::ref(*io_node)));
    }

    while (likely(SignalHandler::get_received_signal() < 0)) {
        std::this_thread::sleep_for(STATS_POLLING_PERIOD);
        for (auto& tracker : m_ipmx_trackers) {
            tracker->print_updates();
        }
    }
    for (auto& thread : m_threads) {
        thread.join();
    }
}

void IPMXReceiverApp::configure_network_flows()
{
    std::string ip_prefix_str;
    int ip_last_octet;
    uint16_t src_port = 0;

    auto ip_vec = CLI::detail::split(m_app_settings->destination_ip, '.');
    ip_prefix_str = std::string(ip_vec[0] + "." + ip_vec[1] + "." + ip_vec[2] + ".");
    ip_last_octet = std::stoi(ip_vec[3]);

    m_data_flows.reserve(m_app_settings->num_of_total_streams);
    m_rtcp_flows.reserve(m_app_settings->num_of_total_streams);
    m_ipmx_trackers.reserve(m_app_settings->num_of_total_streams);

    uint32_t id = 0;
    for (uint32_t flow_index = 0; flow_index < m_app_settings->num_of_total_streams; ++flow_index) {
        std::ostringstream ip;
        ip << ip_prefix_str << (ip_last_octet + flow_index) % IP_OCTET_LEN;
        ReceiveFlow data_flow(id++, m_app_settings->source_ip, src_port, ip.str(),
                              m_app_settings->destination_port, flow_index);
        m_data_flows.push_back(data_flow);
        ReceiveFlow rtcp_flow(id++, m_app_settings->source_ip, src_port, ip.str(),
                              m_app_settings->destination_port + 1, flow_index);
        m_rtcp_flows.push_back(rtcp_flow);
        m_ipmx_trackers.push_back(std::make_shared<IPMXStreamTimelineTracker>(flow_index));
        m_ipmx_trackers.back()->set_statistics_report_interval(m_app_settings->stats_report_interval_ms);
    }
}

void IPMXReceiverApp::initialize_receive_io_nodes()
{
    size_t streams_offset = 0;
    for (size_t rx_idx = 0; rx_idx < m_app_settings->num_of_threads; rx_idx++) {
        int recv_cpu_core = m_app_settings->app_threads_cores[
            rx_idx % m_app_settings->app_threads_cores.size()];

        auto rtp_receiver = std::make_unique<RTPReceiverIONode>(
            *m_app_settings,
            m_ipmx_receiver_settings->is_extended_sequence_number,
            std::vector<std::string>{m_app_settings->local_ip},
            rx_idx,
            recv_cpu_core,
            *m_memory_utils);
        auto flows = std::vector<ReceiveFlow>(
            m_data_flows.begin() + streams_offset,
            m_data_flows.begin() + streams_offset + m_streams_per_thread[rx_idx]);
        initialize_rtp_streams(*rtp_receiver, streams_offset, flows);
        rtp_receiver->set_statistics_report_interval(m_app_settings->stats_report_interval_ms);
        m_receivers.push_back(std::move(rtp_receiver));

        streams_offset += m_streams_per_thread[rx_idx];
    }

    auto rtcp_receiver = std::make_unique<RTPReceiverIONode>(
        m_rtcp_receiver_settings,
        false,
        std::vector<std::string>{m_app_settings->local_ip},
        0,
        m_ipmx_receiver_settings->rtcp_thread_core,
        *m_memory_utils);

    initialize_rtcp_stream(*rtcp_receiver, m_rtcp_flows);
    rtcp_receiver->set_statistics_report_interval(m_app_settings->stats_report_interval_ms);
    m_receivers.push_back(std::move(rtcp_receiver));
}

void IPMXReceiverApp::initialize_rtp_streams(RTPReceiverIONode& node, size_t start_id, const std::vector<ReceiveFlow>& flows)
{
    std::vector<std::unique_ptr<IReceiveStream>> streams;
    streams.reserve(flows.size());
    for (size_t id = start_id; id < start_id + flows.size(); ++id) {
        ReceiveStreamSettings stream_settings(TwoTupleFlow(id, m_app_settings->local_ip, 0),
            RMX_INPUT_APP_PROTOCOL_PACKET,
            RMX_INPUT_TIMESTAMP_SYNCED,
            {RMX_INPUT_STREAM_CREATE_INFO_PER_PACKET},
            m_app_settings->num_of_packets_in_chunk,
            m_app_settings->packet_payload_size,
            m_app_settings->packet_app_header_size);
        auto stream = std::make_unique<AppRTPReceiveStream>(stream_settings,
            m_is_extended_sequence_number,
            m_app_settings->packet_app_header_size != 0);
        stream->set_frame_start_handler(std::make_unique<IPMXFrameStartNotifier>(m_ipmx_trackers[id]));
        streams.push_back(std::move(stream));
    }
    node.assign_streams(start_id, flows, streams);
}

void IPMXReceiverApp::initialize_rtcp_stream(RTPReceiverIONode& node, const std::vector<ReceiveFlow>& flows)
{
    std::vector<std::unique_ptr<IReceiveStream>> streams;
    ReceiveStreamSettings stream_settings(TwoTupleFlow(0, m_app_settings->local_ip, 0),
        RMX_INPUT_APP_PROTOCOL_PACKET,
        RMX_INPUT_TIMESTAMP_SYNCED,
        {RMX_INPUT_STREAM_CREATE_INFO_PER_PACKET},
        m_rtcp_receiver_settings.num_of_packets_in_chunk,
        m_rtcp_receiver_settings.packet_payload_size,
        m_rtcp_receiver_settings.packet_app_header_size);
    auto stream = std::make_unique<AppRTPReceiveStream>(stream_settings, false, false, false);
    streams.push_back(std::move(stream));
    node.assign_streams(0, m_rtcp_flows, streams);
    node.set_receive_data_consumer(0, std::make_unique<RTCPChunkConsumer>(m_ipmx_trackers));
}
