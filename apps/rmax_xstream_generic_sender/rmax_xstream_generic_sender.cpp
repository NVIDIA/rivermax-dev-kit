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

#include <rivermax_api.h>

#include "rt_threads.h"
#include "rmax_xstream_generic_sender.h"
#include "api/rmax_apps_lib_api.h"
#include "io_node/io_node.h"
#include "apps/rmax_base_app.h"

using namespace ral::lib::core;
using namespace ral::lib::services;
using namespace ral::io_node;
using namespace ral::apps::rmax_xstream_generic_sender;


int main(int argc, const char* argv[])
{
    GenericSenderApp app(argc, argv);

    ReturnStatus rc = app.run();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Generic Sender failed to run" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

GenericSenderApp::GenericSenderApp(int argc, const char* argv[]) :
    RmaxBaseApp(APP_DESCRIPTION, APP_EXAMPLES)
{
    m_obj_init_status = initialize(argc, argv);
}

GenericSenderApp::~GenericSenderApp()
{
}

void GenericSenderApp::add_cli_options()
{
    m_cli_parser_manager->add_option(CLIOptStr::LOCAL_IP);
    m_cli_parser_manager->add_option(CLIOptStr::DST_IP);
    m_cli_parser_manager->add_option(CLIOptStr::DST_PORT);
    m_cli_parser_manager->add_option(CLIOptStr::THREADS);
    m_cli_parser_manager->add_option(CLIOptStr::FLOWS);
    m_cli_parser_manager->add_option(CLIOptStr::STREAMS)->check(StreamToThreadsFlowsValidator(
        m_app_settings->num_of_threads, m_app_settings->num_of_total_flows));
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
#endif
    m_cli_parser_manager->add_option(CLIOptStr::ALLOCATOR_TYPE);
}

void GenericSenderApp::post_cli_parse_initialization()
{
    if (m_app_settings->use_checksum_header) {
        m_app_settings->packet_app_header_size = sizeof(ChecksumHeader);
    }
}

ReturnStatus GenericSenderApp::run()
{
    if (m_obj_init_status != ReturnStatus::obj_init_success) {
        return m_obj_init_status;
    }

    try {
        distribute_work_for_threads();
        initialize_send_flows();
        initialize_sender_threads();
        ReturnStatus rc = allocate_app_memory();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to allocate the memory required for the application" << std::endl;
            return rc;
        }
        const int mem_block_indx = 0;
        distribute_memory_for_senders(mem_block_indx);
        run_threads(m_senders);
    }
    catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus GenericSenderApp::initialize_rivermax_resources()
{
    rmax_init_config init_config;
    memset(&init_config, 0, sizeof(init_config));

    if (m_app_settings->internal_thread_core != CPU_NONE) {
        RMAX_CPU_SET(m_app_settings->internal_thread_core, &init_config.cpu_mask);
        init_config.flags |= RIVERMAX_CPU_MASK;
    }
    rt_set_realtime_class();

    init_config.flags |= RIVERMAX_HANDLE_SIGNAL;

    return m_rmax_apps_lib.initialize_rivermax(init_config);
}

ReturnStatus GenericSenderApp::cleanup_rivermax_resources()
{
    rmax_status_t status;

    for (auto& mem_block : m_mem_blocks) {
        status = rmax_deregister_memory(mem_block->mkey_id, m_source_address.sin_addr);
        if (status != RMAX_OK) {
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
    gs_mem_block_t block;

    memset(&block, 0, sizeof(block));
    block.mem_block.pointer = m_mem_allocator->allocate(length);
    block.mem_block.length = length;
    block.mkey_id = 0;

    if (!block.mem_block.pointer) {
        std::cerr << "Failed to allocate application memory" << std::endl;
        return ReturnStatus::failure;
    }

    rmax_status_t status = rmax_register_memory(
        block.mem_block.pointer, block.mem_block.length,
        m_source_address.sin_addr, &block.mkey_id);
    if (status != RMAX_OK) {
        std::cerr << "Failed to register application memory with status: " << status << std::endl;
        return ReturnStatus::failure;
    }

    m_mem_blocks.push_back(std::unique_ptr<gs_mem_block_t>(
        new gs_mem_block_t{ block.mem_block, block.mkey_id }));

    std::cout << "Allocated " << block.mem_block.length <<
        " bytes at address " << block.mem_block.pointer <<
        " with mkey: " << block.mkey_id << std::endl;

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
            m_mem_allocator->get_memory_utils())));
        m_senders[sndr_indx]->initialize_send_flows(flows);
        m_senders[sndr_indx]->initialize_streams(flows_offset);
    }
}

void GenericSenderApp::distribute_memory_for_senders(const int mem_block_index)
{
    auto& app_mem = m_mem_blocks[mem_block_index];
    byte_t* pointer = nullptr;
    rmax_mkey_id mkey = 0;
    size_t length = 0;
    size_t offset = 0;

    for (auto& sender : m_senders) {
        pointer = reinterpret_cast<byte_t*>(app_mem->mem_block.pointer) + offset;
        mkey = app_mem->mkey_id;
        length = sender->initialize_memory(pointer, mkey);
        offset += length;
    }
}
