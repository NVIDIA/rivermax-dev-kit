/*
 * Copyright © 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
#include <cstring>
#include <rivermax_api.h>

#include "rt_threads.h"

#include "apps/rmax_base_app.h"
#include "api/rmax_apps_lib_api.h"

using namespace ral::apps;
using namespace ral::lib::services;

namespace {
static const std::map<AllocatorTypeUI, AllocatorType> UI_ALLOCATOR_TYPE_MAP{
    { AllocatorTypeUI::Auto,            AllocatorType::HugePageDefault },
    { AllocatorTypeUI::HugePageDefault, AllocatorType::HugePageDefault },
    { AllocatorTypeUI::Malloc,          AllocatorType::Malloc },
    { AllocatorTypeUI::HugePage2MB,     AllocatorType::HugePage2MB },
    { AllocatorTypeUI::HugePage512MB,   AllocatorType::HugePage512MB },
    { AllocatorTypeUI::HugePage1GB,     AllocatorType::HugePage1GB }
};
}

RmaxBaseApp::RmaxBaseApp(const std::string& app_description, const std::string& app_examples) :
    m_obj_init_status(ReturnStatus::obj_init_failure),
    m_app_settings(new AppSettings),
    m_rmax_apps_lib(ral::lib::RmaxAppsLibFacade()),
    m_cli_parser_manager(m_rmax_apps_lib.get_cli_parser_manager(
        app_description + rmx_get_version_string(), app_examples, m_app_settings)),
    m_signal_handler(m_rmax_apps_lib.get_signal_handler(true)),
    m_stats_reader(nullptr)
{
    memset(&m_local_address, 0, sizeof(m_local_address));
}

RmaxBaseApp::~RmaxBaseApp()
{
    if (m_obj_init_status != ReturnStatus::obj_init_success) {
        return;
    }

    for (auto& thread : m_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    cleanup_rivermax_resources();
}

void RmaxBaseApp::initialize_common_default_app_settings()
{
    m_app_settings->destination_ip = DESTINATION_IP_DEFAULT;
    m_app_settings->destination_port = DESTINATION_PORT_DEFAULT;
    m_app_settings->num_of_threads = NUM_OF_THREADS_DEFAULT;
    m_app_settings->num_of_total_streams = NUM_OF_TOTAL_STREAMS_DEFAULT;
    m_app_settings->num_of_total_flows = NUM_OF_TOTAL_FLOWS_DEFAULT;
    m_app_settings->internal_thread_core = CPU_NONE;
    m_app_settings->app_threads_cores = std::vector<int>(m_app_settings->num_of_threads, CPU_NONE);
    m_app_settings->rate = { 0, 0 };
    m_app_settings->num_of_chunks = NUM_OF_CHUNKS_DEFAULT;
    m_app_settings->num_of_packets_in_chunk = NUM_OF_PACKETS_IN_CHUNK_DEFAULT;
    m_app_settings->packet_payload_size = PACKET_PAYLOAD_SIZE_DEFAULT;
    m_app_settings->packet_app_header_size = PACKET_APP_HEADER_SIZE_DEFAULT;
    m_app_settings->sleep_between_operations_us = SLEEP_BETWEEN_OPERATIONS_US_DEFAULT;
    m_app_settings->sleep_between_operations = false;
    m_app_settings->print_parameters = false;
    m_app_settings->use_checksum_header = false;
    m_app_settings->hw_queue_full_sleep_us = 0;
    m_app_settings->gpu_id = INVALID_GPU_ID;
    m_app_settings->allocator_type = AllocatorTypeUI::Auto;
    m_app_settings->statistics_reader_core = INVALID_CORE_NUMBER;
    m_app_settings->session_id_stats = UINT_MAX;
}

ReturnStatus RmaxBaseApp::initialize_memory_allocators()
{
    const auto alloc_type_iter = UI_ALLOCATOR_TYPE_MAP.find(m_app_settings->allocator_type);
    if (alloc_type_iter == UI_ALLOCATOR_TYPE_MAP.end()) {
        std::cerr << "Unknown UI allocator type " << static_cast<int>(m_app_settings->allocator_type) << std::endl;
        return ReturnStatus::failure;
    }
    AllocatorType allocator_type = alloc_type_iter->second;
    AllocatorType header_allocator_type;
    AllocatorType payload_allocator_type;
    if (m_app_settings->gpu_id != INVALID_GPU_ID) {
        header_allocator_type = allocator_type;
        payload_allocator_type = AllocatorType::Gpu;
    } else {
        header_allocator_type = allocator_type;
        payload_allocator_type = allocator_type;
    }
    m_header_allocator = m_rmax_apps_lib.get_memory_allocator(header_allocator_type, m_app_settings);
    if (m_header_allocator == nullptr) {
        std::cerr << "Failed to create header memory allocator" << std::endl;
        return ReturnStatus::failure;
    }
    m_payload_allocator = m_rmax_apps_lib.get_memory_allocator(payload_allocator_type, m_app_settings);
    if (m_payload_allocator == nullptr) {
        std::cerr << "Failed to create payload memory allocator" << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

ReturnStatus RmaxBaseApp::initialize(int argc, const char* argv[])
{
    ReturnStatus rc = m_cli_parser_manager->initialize();
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to initialize CLI manager" << std::endl;
        return rc;
    }

    initialize_common_default_app_settings();
    add_cli_options();
    rc = m_cli_parser_manager->parse_cli(argc, argv);
    if (rc == ReturnStatus::failure || rc == ReturnStatus::success_cli_help) {
        return rc;
    }

    post_cli_parse_initialization();

    rc = initialize_memory_allocators();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to initialize memory allocators" << std::endl;
        m_obj_init_status = ReturnStatus::memory_allocation_failure;
        return m_obj_init_status;
    }

    rc = initialize_rivermax_resources();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to initialize Rivermax resources" << std::endl;
        return rc;
    }

    rc = initialize_connection_parameters();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to initialize application connection parameters" << std::endl;
        return rc;
    }

    rc = set_rivermax_clock();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to set Rivermax clock" << std::endl;
        return rc;
    }

    return ReturnStatus::obj_init_success;
}

ReturnStatus RmaxBaseApp::set_rivermax_clock()
{
    return ReturnStatus::success;
}

ReturnStatus RmaxBaseApp::initialize_connection_parameters()
{
    memset(&m_local_address, 0, sizeof(sockaddr_in));
    m_local_address.sin_family = AF_INET;
    int rc = inet_pton(AF_INET, m_app_settings->local_ip.c_str(), &m_local_address.sin_addr);
    if (rc != 1) {
        std::cerr << "Failed to parse local network address: " << m_app_settings->local_ip << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

void RmaxBaseApp::run_stats_reader()
{
    if (!is_run_stats_reader()) {
        return;
    }

    m_stats_reader.reset(new StatisticsReader());
    m_stats_reader->set_cpu_core_affinity(m_app_settings->statistics_reader_core);
    if (m_app_settings->session_id_stats != UINT_MAX) {
        std::cout << "Set presen session id: " << m_app_settings->session_id_stats << std::endl;
        m_stats_reader->set_session_id(m_app_settings->session_id_stats);
    }
    m_threads.push_back(std::thread(std::ref(*m_stats_reader)));
}
