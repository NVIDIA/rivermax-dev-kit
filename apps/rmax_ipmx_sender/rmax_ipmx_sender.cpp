/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "rmax_ipmx_sender.h"
#include "api/rmax_apps_lib_api.h"
#include "io_node/io_node.h"
#include "apps/rmax_base_app.h"
#include "services/utils/clock.h"
#include "services/utils/rtp_video.h"

using namespace ral::lib::core;
using namespace ral::lib::services;
using namespace ral::io_node;
using namespace ral::apps::rmax_ipmx_sender;


int main(int argc, const char* argv[])
{
    IpmxSenderApp app(argc, argv);

    ReturnStatus rc = app.run();
    if (rc == ReturnStatus::failure) {
        std::cerr << "IPMX Sender failed to run" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

IpmxSenderApp::IpmxSenderApp(int argc, const char* argv[]) :
    RmaxBaseApp(APP_DESCRIPTION, APP_EXAMPLES),
    m_device_interface{0},
    m_mem_region{nullptr, 0, 0}
{
    m_obj_init_status = initialize(argc, argv);
}

IpmxSenderApp::~IpmxSenderApp()
{
}

void IpmxSenderApp::initialize_common_default_app_settings()
{
    RmaxBaseApp::initialize_common_default_app_settings();
    m_app_settings->ref_clk_is_ptp = false;
}

void IpmxSenderApp::add_cli_options()
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
    auto parser = m_cli_parser_manager->get_parser();
    parser->add_option(
        "-x,--stream-type",
        m_app_settings->video_stream_type,
        "Type of a video stream")->check(CLI::IsMember(SUPPORTED_STREAMS))->required();
    // TODO: move PTP flag to App
    parser->add_flag("--ptp", m_app_settings->ref_clk_is_ptp,
                     "Use NIC RTC as a PTP-synchronized Common Reference clock");
    auto stats_core = parser->add_option(
        "-R,--statistics-core",
        m_app_settings->statistics_reader_core,
        "CPU core affinity for statistics reader thread")->check(CLI::Range(0, MAX_CPU_RANGE));
    parser->add_option(
        "-P,--session-id-stats",
        m_app_settings->session_id_stats,
        "Present runtime statistics of the given session id")->check(
            CLI::PositiveNumber)->needs(stats_core);
}

void IpmxSenderApp::post_cli_parse_initialization()
{
    m_app_settings->media.frames_fields_in_mem_block = MIN_FRAMES_FOR_SIMULTANEOUS_TX_AND_FILLUP;
    compose_ipmx_media_settings(*m_app_settings);
}

ReturnStatus IpmxSenderApp::run()
{
    if (m_obj_init_status != ReturnStatus::obj_init_success) {
        return m_obj_init_status;
    }

    try {
        assign_streams_to_threads();
        initialize_send_flows();
        initialize_sender_threads();
        ReturnStatus rc = allocate_app_memory();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to allocate the memory required for the application" << std::endl;
            return rc;
        }
        distribute_memory_to_senders();
        run_stats_reader();
        run_threads(m_senders);
    }
    catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus IpmxSenderApp::initialize_rivermax_resources()
{
    rt_set_realtime_class();
    return m_rmax_apps_lib.initialize_rivermax(m_app_settings->internal_thread_core);
}

ReturnStatus IpmxSenderApp::initialize_connection_parameters()
{
    ReturnStatus rc = RmaxBaseApp::initialize_connection_parameters();
    if (rc != ReturnStatus::success) {
        return rc;
    }
    return init_device_iface(m_device_interface);
}

ReturnStatus IpmxSenderApp::init_device_iface(rmx_device_iface& device_iface)
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

ReturnStatus IpmxSenderApp::set_rivermax_clock()
{
    std::cout << "Switching to PTP clock" << std::endl;
    ReturnStatus rc = set_rivermax_ptp_clock(&m_device_interface);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    TimeContext& tc = TimeContext::get();
    return tc.init(m_app_settings->ref_clk_is_ptp);
}

void IpmxSenderApp::initialize_send_flows()
{
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

void IpmxSenderApp::assign_streams_to_threads()
{
    m_streams_per_thread.reserve(m_app_settings->num_of_threads);
    for (int stream = 0; stream < m_app_settings->num_of_total_streams; stream++) {
        m_streams_per_thread[stream % m_app_settings->num_of_threads]++;
    }
}

void IpmxSenderApp::initialize_sender_threads()
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
        m_senders.push_back(std::make_unique<IpmxSenderIONode>(
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

size_t IpmxSenderApp::query_memory_size()
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

void* IpmxSenderApp::allocate_and_align_payload(size_t size)
{
    size = m_payload_allocator->align_length(size);
    return m_payload_allocator->allocate_aligned(size, m_payload_allocator->get_page_size());
}

ReturnStatus IpmxSenderApp::allocate_app_memory()
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

void IpmxSenderApp::distribute_memory_to_senders()
{
    byte_t* pointer = reinterpret_cast<byte_t*>(m_mem_region.addr);
    rmx_mkey_id mkey = m_mem_region.mkey;
    for (auto& sender : m_senders) {
        size_t length = sender->initialize_memory(pointer, mkey);
        pointer += length;
    }
}
