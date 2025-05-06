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

#include "rdk/apps/rmax_xstream_media_sender/rmax_xstream_media_sender.h"
#include "rdk/apps/rmax_base_memory_strategy.h"
#include "rdk/services/utils/defs.h"
#include "rdk/services/utils/clock.h"
#include "rdk/services/utils/enum_utils.h"
#include "rdk/services/sdp/sdp_defs.h"
#include "rdk/services/media/media.h"

using namespace rivermax::dev_kit::apps::rmax_xstream_media_sender;

void MediaSenderSettings::init_default_values()
{
    AppSettings::init_default_values();
    media.frames_fields_in_mem_block = 1;
}

ReturnStatus MediaSenderSettingsValidator::validate(const std::shared_ptr<MediaSenderSettings>& settings) const
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

ReturnStatus MediaSenderCLISettingsBuilder::add_cli_options(std::shared_ptr<MediaSenderSettings>& settings)
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
    auto hds = m_cli_parser_manager->add_option(CLIOptStr::HEADER_DATA_SPLIT);
#ifdef CUDA_ENABLED
    m_cli_parser_manager->add_option(CLIOptStr::GPU_ID)->needs(hds);
    m_cli_parser_manager->add_option(CLIOptStr::LOCK_GPU_CLOCKS);
#endif
    auto stats_enabled = m_cli_parser_manager->add_option(CLIOptStr::ENABLE_STATS_READER);
    auto stats_core = m_cli_parser_manager->add_option(CLIOptStr::STATS_CORE)->needs(stats_enabled);
    m_cli_parser_manager->add_option(CLIOptStr::STATS_SESSION_ID)->needs(stats_enabled);
    auto mem = m_cli_parser_manager->add_option(CLIOptStr::APP_MEMORY_ALLOC);
    m_cli_parser_manager->add_option(CLIOptStr::ALLOCATOR_TYPE)->needs(mem);
    m_cli_parser_manager->add_option(CLIOptStr::REGISTER_MEMORY)->needs(mem);
    auto video_file = m_cli_parser_manager->add_option(CLIOptStr::VIDEO_FILE)->needs(mem);
    m_cli_parser_manager->add_option(CLIOptStr::VIDEO_RESOLUTION)
        ->group(CLIGroupStr::VIDEO_FORMAT_OPTIONS);
    m_cli_parser_manager->add_option(CLIOptStr::VIDEO_FRAME_RATE)
        ->group(CLIGroupStr::VIDEO_FORMAT_OPTIONS);
    m_cli_parser_manager->add_option(CLIOptStr::VIDEO_SAMPLING)
        ->group(CLIGroupStr::VIDEO_FORMAT_OPTIONS);
    m_cli_parser_manager->add_option(CLIOptStr::VIDEO_BIT_DEPTH)
        ->group(CLIGroupStr::VIDEO_FORMAT_OPTIONS);
    m_cli_parser_manager->add_option(CLIOptStr::DYNAMIC_FILE_LOADING)->needs(video_file);

    return ReturnStatus::success;
}

MediaSenderApp::MediaSenderApp(std::shared_ptr<ISettingsBuilder<MediaSenderSettings>> settings_builder) :
    RmaxBaseApp(),
    m_settings_builder(std::move(settings_builder)),
    m_device_interface{}
{
}

ReturnStatus MediaSenderApp::post_load_settings()
{
    auto rc = initialize_media_settings(*m_app_settings);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to initialize media settings" << std::endl;
    }
    return rc;
}

ReturnStatus MediaSenderApp::initialize_app_settings()
{
    if (m_settings_builder == nullptr) {
        std::cerr << "Settings builder is not initialized" << std::endl;
        return ReturnStatus::failure;
    }
    m_media_sender_settings = std::make_shared<MediaSenderSettings>();
    ReturnStatus rc = m_settings_builder->build(m_media_sender_settings);
    if (rc == ReturnStatus::success) {
        m_app_settings = m_media_sender_settings;
        return ReturnStatus::success;
    }
    if (rc != ReturnStatus::success_cli_help) {
        std::cerr << "Failed to build settings" << std::endl;
    }
    m_obj_init_status = rc;
    return rc;
}

ReturnStatus MediaSenderApp::initialize()
{
    ReturnStatus rc  = RmaxBaseApp::initialize();

    if (rc != ReturnStatus::obj_init_success) {
        return m_obj_init_status;
    }

    try {
        distribute_work_for_threads();
        configure_network_flows();
        initialize_sender_threads();
        rc = configure_memory_layout();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to configure memory layout" << std::endl;
            return rc;
        }
        rc = set_internal_frame_providers();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to set internal frame providers" << std::endl;
            return rc;
        }
    }
    catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return ReturnStatus::failure;
    }

    m_obj_init_status = ReturnStatus::obj_init_success;
    return m_obj_init_status;
}

ReturnStatus MediaSenderApp::initialize_connection_parameters()
{
    in_addr device_address;
    if (inet_pton(AF_INET, m_app_settings->local_ip.c_str(), &device_address) != 1) {
        std::cerr << "Failed to parse address of device " << m_app_settings->local_ip << std::endl;
        return ReturnStatus::failure;
    }
    rmx_status status = rmx_retrieve_device_iface_ipv4(&m_device_interface, &device_address);
    if (status != RMX_OK) {
        std::cerr << "Failed to get device: " << m_app_settings->local_ip << " with status: " << status << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus MediaSenderApp::initialize_memory_strategy()
{
    std::vector<rmx_device_iface> device_interfaces = {m_device_interface};
    auto base_memory_strategy = std::make_unique<RmaxBaseMemoryStrategy>(
        *m_header_allocator, *m_payload_allocator,
        *m_memory_utils,
        device_interfaces,
        m_num_paths_per_stream,
        m_app_settings->app_memory_alloc,
        m_app_settings->register_memory);

    for (const auto& sender : m_senders) {
        base_memory_strategy->add_memory_subcomponent(sender);
    }

    m_memory_strategy.reset(base_memory_strategy.release());

    return ReturnStatus::success;
}

ReturnStatus MediaSenderApp::run()
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

ReturnStatus MediaSenderApp::set_rivermax_clock()
{
    ReturnStatus rc = set_rivermax_ptp_clock(&m_device_interface);
    if(rc == ReturnStatus::success) {
        uint64_t ptp_time = 0;
        rc = get_rivermax_ptp_time_ns(ptp_time);
    }
    return rc;
}

void MediaSenderApp::configure_network_flows()
{
    // TODO: Make this controllable from application level.
    constexpr bool dest_port_iteration = false;
    auto ip_vec = CLI::detail::split(m_app_settings->destination_ip, '.');
    auto ip_prefix_str = std::string(ip_vec[0] + "." + ip_vec[1] + "." + ip_vec[2] + ".");
    auto ip_last_octet = std::stoi(ip_vec[3]);
    size_t flow_index = 0;
    std::ostringstream ip;
    uint16_t port;

    m_flows.reserve(m_app_settings->num_of_total_flows);
    while (flow_index != m_app_settings->num_of_total_flows) {
        if (dest_port_iteration) {
            ip << m_app_settings->destination_ip;
            port = m_app_settings->destination_port + static_cast<uint16_t>(flow_index);
        } else {
            ip << ip_prefix_str << (ip_last_octet + flow_index) % IP_OCTET_LEN;
            port = m_app_settings->destination_port;
        }

        m_flows.push_back(TwoTupleFlow(flow_index, ip.str(), port));
        ip.str("");
        flow_index++;
    }
}

void MediaSenderApp::distribute_work_for_threads()
{
    m_app_settings->num_of_threads = std::min<size_t>(m_app_settings->num_of_threads, m_app_settings->num_of_total_streams);
    m_streams_per_thread.reserve(m_app_settings->num_of_threads);
    for (int stream = 0; stream < m_app_settings->num_of_total_streams; stream++) {
        m_streams_per_thread[stream % m_app_settings->num_of_threads]++;
    }
}

void MediaSenderApp::initialize_sender_threads()
{
    size_t streams_offset = 0;
    for (size_t sndr_indx = 0; sndr_indx < m_app_settings->num_of_threads; sndr_indx++) {
        int sender_cpu_core;
        if (sndr_indx < m_app_settings->app_threads_cores.size()) {
            sender_cpu_core = m_app_settings->app_threads_cores[sndr_indx];
        } else {
            std::cerr << "Warning: CPU afinity for Sender " << sndr_indx <<
                         " is not set!!!" << std::endl;
            sender_cpu_core = CPU_NONE;
        }
        auto network_address = FourTupleFlow(
            sndr_indx,
            m_app_settings->local_ip,
            m_app_settings->source_port,
            m_app_settings->destination_ip,
            m_app_settings->destination_port);
        auto flows = std::vector<TwoTupleFlow>(
            m_flows.begin() + streams_offset,
            m_flows.begin() + streams_offset + m_streams_per_thread[sndr_indx]);
        m_senders.push_back(std::unique_ptr<MediaSenderIONode>(new MediaSenderIONode(
            network_address,
            m_app_settings,
            sndr_indx,
            m_streams_per_thread[sndr_indx],
            sender_cpu_core,
            *m_memory_utils,
            MediaSenderApp::get_time_ns)));
        m_senders[sndr_indx]->initialize_send_flows(flows);
        m_senders[sndr_indx]->initialize_streams();
        streams_offset += m_streams_per_thread[sndr_indx];
    }
}

ReturnStatus MediaSenderApp::set_frame_provider(size_t stream_index,
    std::shared_ptr<IFrameProvider> frame_provider, MediaType media_type, bool contains_payload)
{
    size_t sender_thread_index = 0;
    size_t sender_stream_index = 0;

    auto rc = find_internal_stream_index(stream_index, sender_thread_index, sender_stream_index);
    if (rc != ReturnStatus::success) {
        std::cerr << "Error setting frame provider, invalid stream index " << stream_index << std::endl;
        return rc;
    }

    rc = m_senders[sender_thread_index]->set_frame_provider(
        sender_stream_index, std::move(frame_provider), media_type, contains_payload);

    if (rc != ReturnStatus::success) {
        std::cerr << "Error setting frame provider for stream "
                  << sender_stream_index << " on sender " << sender_thread_index << std::endl;
    }

    return rc;
}

ReturnStatus MediaSenderApp::set_internal_frame_providers()
{
    std::shared_ptr<IFrameProvider> frame_provider;
    ReturnStatus rc;
    bool contains_payload = true;
    for (size_t sender_index = 0; sender_index < m_app_settings->num_of_threads; sender_index++) {
        const size_t streams_in_thread = m_streams_per_thread[sender_index];
        for(size_t stream_index = 0; stream_index < streams_in_thread; stream_index++) {
            if (m_app_settings->video_file.empty() || !(m_app_settings->dynamic_video_file_load)) {
                frame_provider = std::make_shared<NullFrameProvider>(m_app_settings->media);
                contains_payload = false;
            } else {
                frame_provider = std::make_shared<MediaFileFrameProvider>(
                    m_app_settings->video_file, MediaType::Video,
                    m_app_settings->media.bytes_per_frame, *m_header_allocator, true);
                auto media_file_frame_provider = std::dynamic_pointer_cast<MediaFileFrameProvider>(frame_provider);
                rc = media_file_frame_provider->load_frames();
                if (rc != ReturnStatus::success) {
                    std::cerr << "Failed to load frames from video file" << std::endl;
                    return rc;
                }
            }
            rc = m_senders[sender_index]->set_frame_provider(
                stream_index, std::move(frame_provider), MediaType::Video, contains_payload);
            if (rc != ReturnStatus::success) {
                std::cerr << "Error setting frame provider for stream "
                          << stream_index << " on sender " << sender_index << std::endl;
                return rc;
            }
        }
    }
    return ReturnStatus::success;
}

uint64_t MediaSenderApp::get_time_ns(void* context)
{
    NOT_IN_USE(context);
    uint64_t ptp_time = 0;
    ReturnStatus rc = get_rivermax_ptp_time_ns(ptp_time);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to get PTP time" << std::endl;
        return 0;
    }
    return ptp_time;
}
