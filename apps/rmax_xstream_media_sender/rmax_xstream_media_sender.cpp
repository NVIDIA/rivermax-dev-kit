/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <string>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

#include <rivermax_api.h>

#include "rt_threads.h"
#include "rmax_xstream_media_sender.h"
#include "api/rmax_apps_lib_api.h"
#include "io_node/io_node.h"
#include "apps/rmax_base_app.h"
#include "services/utils/clock.h"
#include "services/utils/rtp_video.h"

using namespace ral::lib::core;
using namespace ral::lib::services;
using namespace ral::io_node;
using namespace ral::apps::rmax_xstream_media_sender;


int main(int argc, const char* argv[])
{
    MediaSenderApp app(argc, argv);

    ReturnStatus rc = app.run();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Media Sender failed to run" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

MediaSenderApp::MediaSenderApp(int argc, const char* argv[]) :
    RmaxBaseApp(APP_DESCRIPTION, APP_EXAMPLES)
{
    m_obj_init_status = initialize(argc, argv);
}

MediaSenderApp::~MediaSenderApp()
{
}

void MediaSenderApp::add_cli_options()
{
    m_cli_parser_manager->add_option(CLIOptStr::LOCAL_IP);
    m_cli_parser_manager->add_option(CLIOptStr::DST_IP);
    m_cli_parser_manager->add_option(CLIOptStr::DST_PORT);
    m_cli_parser_manager->add_option(CLIOptStr::THREADS);
    m_cli_parser_manager->add_option(CLIOptStr::STREAMS)->check(
        StreamToThreadsValidator(m_app_settings->num_of_threads));
    m_cli_parser_manager->add_option(CLIOptStr::VERBOSE);
    m_cli_parser_manager->add_option(CLIOptStr::INTERNAL_CORE);
    m_cli_parser_manager->add_option(CLIOptStr::APPLICATION_CORE);
    m_cli_parser_manager->add_option(CLIOptStr::SLEEP);
    m_cli_parser_manager->add_option(CLIOptStr::ALLOCATOR_TYPE);
    // TODO: Remove this after adding SDP parser, add SDP file parameter.
    m_cli_parser_manager->get_parser()->add_option(
        "-x,--stream-type",
        m_app_settings->video_stream_type,
        "Type of a video stream")->check(CLI::IsMember(SUPPORTED_STREAMS))->required();
    auto stats_core = m_cli_parser_manager->get_parser()->add_option(
        "-R,--statistics-core",
        m_app_settings->statistics_reader_core,
        "CPU core affinity for statistics reader thread")->check(CLI::Range(0, MAX_CPU_RANGE));
    m_cli_parser_manager->get_parser()->add_option(
        "-P,--session-id-stats",
        m_app_settings->session_id_stats,
        "Present runtime statistics of the given session id")->check(
            CLI::PositiveNumber)->needs(stats_core);
}

void MediaSenderApp::post_cli_parse_initialization()
{
    /* TODO - Remove this code and add the following:
     *    * Get SDP as an argument
     *    * Use SDP parser
     *    * Generalize the code as service/core component in the library
    */
    m_app_settings->media.frames_fields_in_mem_block = 1;
    compose_media_settings(*m_app_settings);
}

ReturnStatus MediaSenderApp::run()
{
    if (m_obj_init_status != ReturnStatus::obj_init_success) {
        return m_obj_init_status;
    }

    try {
        distribute_work_for_threads();
        initialize_send_flows();
        initialize_sender_threads();
        run_stats_reader();
        run_threads(m_senders);
    }
    catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus MediaSenderApp::initialize_rivermax_resources()
{
    rt_set_realtime_class();
    return m_rmax_apps_lib.initialize_rivermax(m_app_settings->internal_thread_core);
}

ReturnStatus MediaSenderApp::set_rivermax_clock()
{
    return set_rivermax_user_clock(MediaSenderApp::get_time_ns);
}

void MediaSenderApp::initialize_send_flows()
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
            m_payload_allocator->get_memory_utils(),
            MediaSenderApp::get_time_ns)));
        m_senders[sndr_indx]->initialize_memory();
        m_senders[sndr_indx]->initialize_send_flows(flows);
        m_senders[sndr_indx]->initialize_streams();
        streams_offset += m_streams_per_thread[sndr_indx];
    }
}

uint64_t MediaSenderApp::get_time_ns(void* context)
{
    NOT_IN_USE(context);
    auto tai_time_now = (std::chrono::system_clock::now() + std::chrono::seconds{ LEAP_SECONDS }).time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(tai_time_now).count();
}
