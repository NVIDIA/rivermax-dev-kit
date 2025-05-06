/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>

#include <rivermax_api.h>

#include "rt_threads.h"
#include "rmax_rtp_receiver.h"
#include "api/rmax_apps_lib_api.h"
#include "io_node/io_node.h"
#include "apps/rmax_base_app.h"

using namespace ral::lib::core;
using namespace ral::lib::services;
using namespace ral::io_node;
using namespace ral::apps::rmax_rtp_receiver;

int main(int argc, const char* argv[])
{
    RTPReceiverApp app(argc, argv);

    ReturnStatus rc = app.run();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Rivermax RTP Receiver failed to run" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

RTPReceiverApp::RTPReceiverApp(int argc, const char* argv[]) :
    RmaxReceiverBaseApp(APP_DESCRIPTION, APP_EXAMPLES)
{
    m_obj_init_status = initialize(argc, argv);
}

void RTPReceiverApp::add_cli_options()
{
    // set application-specific default values
    m_app_settings->num_of_packets_in_chunk = DEFAULT_NUM_OF_PACKETS_IN_CHUNK;

    // set CLI options
    std::shared_ptr<CLI::App> parser = m_cli_parser_manager->get_parser();

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

    parser->add_flag("-X,--ext-seq-num", m_is_extended_sequence_number, "Parse extended sequence number from RTP payload");
}

ReturnStatus RTPReceiverApp::initialize_connection_parameters()
{
    in_addr device_address;
    if (inet_pton(AF_INET, m_app_settings->local_ip.c_str(), &device_address) != 1) {
        std::cerr << "Failed to parse address of device " << m_app_settings->local_ip << std::endl;
        return ReturnStatus::failure;
    }
    rmx_status status = rmx_retrieve_device_iface_ipv4(&m_device_iface, &device_address);
    if (status != RMX_OK) {
        std::cerr << "Failed to get device: " << m_app_settings->local_ip << " with status: " << status << std::endl;
        return ReturnStatus::failure;
    }

#if defined(CUDA_ENABLED) && !defined(TEGRA_ENABLED)
    if (m_app_settings->gpu_id != INVALID_GPU_ID && m_app_settings->packet_app_header_size == 0) {
        std::cerr << "GPU Direct is supported only in header-data split mode!\n"
                << "Please specify application header size with --app-hdr-size option" << std::endl;
        return ReturnStatus::failure;
    }
#endif

    return ReturnStatus::success;
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
        FourTupleFlow path(id++, m_app_settings->source_ip, src_port, ip.str(), m_app_settings->destination_port);
        m_flows.push_back(path);
    }
}

void RTPReceiverApp::initialize_receive_io_nodes()
{
    size_t streams_offset = 0;
    for (size_t rx_idx = 0; rx_idx < m_app_settings->num_of_threads; rx_idx++) {
        int recv_cpu_core = m_app_settings->app_threads_cores[rx_idx % m_app_settings->app_threads_cores.size()];

        auto flows = std::vector<FourTupleFlow>(
            m_flows.begin() + streams_offset,
            m_flows.begin() + streams_offset + m_streams_per_thread[rx_idx]);
        m_receivers.push_back(std::unique_ptr<RTPReceiverIONode>(new RTPReceiverIONode(
            *m_app_settings,
            m_is_extended_sequence_number,
            m_app_settings->local_ip,
            rx_idx,
            recv_cpu_core)));
        m_receivers[rx_idx]->initialize_streams(streams_offset, flows);
        streams_offset += m_streams_per_thread[rx_idx];
    }
}

ReturnStatus RTPReceiverApp::register_app_memory()
{
    if (!m_app_settings->register_memory) {
        return ReturnStatus::success;
    }

    m_header_mem_region.addr = m_header_buffer;
    m_header_mem_region.length = m_header_mem_size;
    m_header_mem_region.mkey = 0;
    if (m_header_mem_size) {
        rmx_mem_reg_params mem_registry;
        rmx_init_mem_registry(&mem_registry, &m_device_iface);
        rmx_status status = rmx_register_memory(&m_header_mem_region, &mem_registry);
        if (status != RMX_OK) {
            std::cerr << "Failed to register header memory on device " << m_app_settings->local_ip
                    << " with status: " << status << std::endl;
            return ReturnStatus::failure;
        }
    }

    rmx_mem_reg_params mem_registry;
    rmx_init_mem_registry(&mem_registry, &m_device_iface);
    m_payload_mem_region.addr = m_payload_buffer;
    m_payload_mem_region.length = m_payload_mem_size;
    rmx_status status = rmx_register_memory(&m_payload_mem_region, &mem_registry);
    if (status != RMX_OK) {
        std::cerr << "Failed to register payload memory on device " << m_app_settings->local_ip
                << " with status: " << status << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

void RTPReceiverApp::unregister_app_memory()
{
    if (!m_app_settings->register_memory) {
        return;
    }

    if (m_header_buffer) {
        rmx_status status = rmx_deregister_memory(&m_header_mem_region, &m_device_iface);
        if (status != RMX_OK) {
            std::cerr << "Failed to deregister header memory on device " << m_app_settings->local_ip
                    << " with status: " << status << std::endl;
        }
    }
    rmx_status status = rmx_deregister_memory(&m_payload_mem_region, &m_device_iface);
    if (status != RMX_OK) {
        std::cerr << "Failed to deregister payload memory on device " << m_app_settings->local_ip
                << " with status: " << status << std::endl;
    }
}

ReturnStatus RTPReceiverApp::get_total_streams_memory_size(size_t& hdr_mem_size, size_t& pld_mem_size)
{
    hdr_mem_size = 0;
    pld_mem_size = 0;
    m_buffer_sizes.clear();
    m_buffer_sizes.reserve(m_receivers.size());

    for (const auto& receiver : m_receivers) {
        for (const auto& stream : receiver->get_streams()) {
            size_t hdr_buf_size, pld_buf_size;
            ReturnStatus rc = stream->query_buffer_size(hdr_buf_size, pld_buf_size);
            if (rc != ReturnStatus::success) {
                std::cerr << "Failed to query buffer size for stream " << stream->get_id()
                        << " of receiver " << receiver->get_index() << std::endl;
                return rc;
            }
            hdr_buf_size = m_header_allocator->align_length(hdr_buf_size);
            pld_buf_size = m_payload_allocator->align_length(pld_buf_size);
            hdr_mem_size += hdr_buf_size;
            pld_mem_size += pld_buf_size;
            m_buffer_sizes.push_back(std::make_pair(hdr_buf_size, pld_buf_size));
        }
    }

    std::cout << "Application requires " << hdr_mem_size << " bytes of header memory and "
        << pld_mem_size << " bytes of payload memory" << std::endl;

    return ReturnStatus::success;
}

void RTPReceiverApp::distribute_memory_for_receivers()
{
    byte_t* hdr_ptr = m_header_buffer;
    byte_t* pld_ptr = m_payload_buffer;

    assert(m_buffer_sizes.size() == m_receivers.size());
    for (size_t i = 0; i < m_receivers.size(); ++i) {
        auto& receiver = m_receivers[i];
        for (auto& stream : receiver->get_streams()) {
            const auto& sizes = m_buffer_sizes[i];

            stream->set_buffers(hdr_ptr, pld_ptr);
            if (m_app_settings->register_memory) {
                stream->set_memory_keys(m_header_mem_region.mkey, m_payload_mem_region.mkey);
            }
            if (hdr_ptr) {
                hdr_ptr += sizes.first;
            }
            pld_ptr += sizes.second;
        }
    }
}
