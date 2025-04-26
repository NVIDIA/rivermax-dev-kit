/*
 * Copyright © 2017-2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#include <string>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

#include <rivermax_api.h>

#include "rt_threads.h"
#include "rmax_xstream_media_sender.h"
#include "api/rmax_apps_lib_api.h"
#include "io_node/io_node.h"
#include "apps/rmax_base_app.h"

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
    // TODO: Remove this after adding SDP parser, add SDP file parameter.
    m_cli_parser_manager->get_parser()->add_option(
        "-x,--stream-type",
        m_app_settings->video_stream_type,
        "Type of a video stream")->check(CLI::IsMember(SUPPORTED_STREAMS))->required();
}

void MediaSenderApp::post_cli_parse_initialization()
{
    /* TODO - Remove this code and add the following:
     *    * Get SDP as an argument
     *    * Use SDP parser
     *    * Generalize the code as service/core component in the library
    */
    auto& s = m_app_settings;
    m_app_settings->num_of_total_flows = m_app_settings->num_of_total_streams;

    if (s->video_stream_type.compare(VIDEO_2110_20_1080p60) == 0) {
        s->media.resolution = { FHD_WIDTH, FHD_HEIGHT };
    } else if (s->video_stream_type.compare(VIDEO_2110_20_2160p60) == 0) {
        s->media.resolution = { UHD_WIDTH, UHD_HEIGHT };
    }

    std::stringstream sdp;
    sdp << "v=0\n"
        << "o=- 1443716955 1443716955 IN IP4 " << s->source_ip << "\n"
        << "s=SMPTE ST2110-20 narrow gap " << s->video_stream_type << "\n"
        << "t=0 0\n"
        << "m=video " << s->destination_port << " RTP/AVP 96\n"
        << "c=IN IP4 " << s->destination_ip << "/64\n"
        << "a=source-filter: incl IN IP4 " << s->destination_ip << " " << s->source_ip << "\n"
        << "a=rtpmap:96 raw/90000\n"
        << "a=fmtp:96 sampling=YCbCr-4:2:2; width="
        << s->media.resolution.width << "; height=" << s->media.resolution.height << "; exactframerate=60; depth=10;"
        << " TCS=SDR; colorimetry=BT709; PM=2110GPM; SSN=ST2110-20:2017; TP=2110TPN;\n"
        << "a=mediaclk:direct=0\n"
        << "a=ts-refclk:localmac=40-a3-6b-a0-2b-d2";

    s->media.sdp = sdp.str();
    s->media.media_block_index = 0;
    s->media.stream_type = StreamType::Video2110_20;
    s->media.video_scan_type = VideoScanType::Progressive;
    s->media.sample_rate = 90000;
    s->media.frame_rate = { 60, 1 };
    s->media.tp_mode = TPMode::TPN;
    s->media.packets_in_frame_field = HD_PACKETS_PER_FRAME_422_10B * \
        (s->media.resolution.width / FHD_WIDTH) * \
        (s->media.resolution.height / FHD_HEIGHT);

    s->num_of_memory_blocks = 1;
    s->packet_payload_size = 1220;  // Including RTP header.

    const size_t lines_in_chunk = 4;
    s->media.packets_in_line = s->media.packets_in_frame_field / s->media.resolution.height;
    s->num_of_packets_in_chunk = lines_in_chunk * s->media.packets_in_line;

    s->media.frame_field_time_interval_ns = NS_IN_SEC / static_cast<double>(
        s->media.frame_rate.num / s->media.frame_rate.denom);
    s->media.lines_in_frame_field = s->media.resolution.height;

    if (s->media.video_scan_type == VideoScanType::Interlaced) {
        s->media.packets_in_frame_field /= 2;
        s->media.lines_in_frame_field /= 2;
        s->media.frame_field_time_interval_ns /= 2;
    }

    s->media.chunks_in_frame_field = static_cast<size_t>(std::ceil(
        s->media.packets_in_frame_field / static_cast<double>(s->num_of_packets_in_chunk)));
    s->media.frames_fields_in_mem_block = 1;
    s->num_of_chunks_in_mem_block = s->media.frames_fields_in_mem_block * s->media.chunks_in_frame_field;
    s->num_of_packets_in_mem_block = s->num_of_chunks_in_mem_block * s->num_of_packets_in_chunk;
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
    rmax_init_config init_config;
    memset(&init_config, 0, sizeof(init_config));

    init_config.flags |= RIVERMAX_HANDLE_SIGNAL;

    if (m_app_settings->internal_thread_core != CPU_NONE) {
        RMAX_CPU_SET(m_app_settings->internal_thread_core, &init_config.cpu_mask);
        init_config.flags |= RIVERMAX_CPU_MASK;
    }
    rt_set_realtime_class();
    return m_rmax_apps_lib.initialize_rivermax(init_config);
}

ReturnStatus MediaSenderApp::set_rivermax_clock()
{
    rmax_clock_t clock;
    memset(&clock, 0, sizeof(clock));

    clock.clock_type = rmax_clock_types::RIVERMAX_USER_CLOCK_HANDLER;
    clock.clock_u.rmax_user_clock_handler.clock_handler = MediaSenderApp::get_time_ns;
    clock.clock_u.rmax_user_clock_handler.ctx = nullptr;

    return m_rmax_apps_lib.set_rivermax_clock(clock);
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
        int sender_cpu_core = m_app_settings->app_threads_cores[sndr_indx % m_app_settings->app_threads_cores.size()];
        auto network_address = FourTupleFlow(
            sndr_indx,
            m_app_settings->source_ip,
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
            m_mem_allocator->get_memory_utils(),
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
