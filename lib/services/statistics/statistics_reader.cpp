/*
 * Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#include <climits>
#include <map>
#include <thread>
#include <string.h>
#include <rmx_stats_api.h>

#include "rt_threads.h"
#include "services/statistics/statistics_reader.h"
#include "services/utils/utils.h"
#include "services/utils/cpu.h"

using namespace ral::lib::services;

typedef std::map<rmx_stats_type, const char*> stats_type_name_map_t;

namespace {
static constexpr uint32_t DEFAULT_SESSION_ID = 0;
static constexpr uint32_t DEFAULT_PROCESS_ID = 0;
static constexpr uint32_t DEFAULT_UPDATE_TIME = 0;

static stats_type_name_map_t stats_type_to_name = {
    { RMX_STATS_SESSION_START, "session_start" },
    { RMX_STATS_SESSION_STOP, "session_stop" },
    { RMX_STATS_SESSION_RUN, "session_run" },
    { RMX_STATS_TX_QUEUE, "tx_queue" },
    { RMX_STATS_RX_QUEUE, "rx_queue" },
    { RMX_STATS_TIME, "time" }
};
}

statistics_handler_cb_map_t StatisticsReader::s_statistics_handler_cb_map = {
        { RMX_STATS_SESSION_START, &StatisticsReader::handle_session_start },
        { RMX_STATS_SESSION_STOP, &StatisticsReader::handle_session_stop },
        { RMX_STATS_SESSION_RUN, &StatisticsReader::handle_session_run },
        { RMX_STATS_TX_QUEUE, &StatisticsReader::handle_tx_queue },
        { RMX_STATS_RX_QUEUE, &StatisticsReader::handle_rx_queue },
        { RMX_STATS_TIME, &StatisticsReader::handle_unknown }
};

StatisticsReader::StatisticsReader() :
    m_stats_config(),
    m_stats_message(),
    m_stats_consumer(),
    m_process_id(DEFAULT_PROCESS_ID),
    m_stats_consumer_created(false),
    m_thread_id(RMX_STATS_INVALID_THREAD_ID),
    m_session_id(DEFAULT_SESSION_ID),
    m_cpu_core_affinity(CPU_NONE),
    m_update_time_sec(DEFAULT_UPDATE_TIME),
    m_statistics_handler_cb_map(s_statistics_handler_cb_map)
{
}

ReturnStatus StatisticsReader::set_stats_config()
{
    rmx_stats_config_set_process_id(&m_stats_config, m_process_id);

    auto status = rmx_stats_config_register_stats_type(&m_stats_config, RMX_STATS_SESSION_RUN);
    if (status != RMX_OK) {
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

ReturnStatus StatisticsReader::init_stats_reader()
{
    rmx_stats_init_config(&m_stats_config);

    if (set_stats_config() != ReturnStatus::success) {
        return ReturnStatus::failure;
    }

    rmx_stats_init_consumer(&m_stats_consumer, &m_stats_config);

    auto status = rmx_stats_create_consumer(&m_stats_consumer, &m_stats_config);
    if (status == RMX_OK) {
        m_stats_consumer_created = true;
    } else {
        return ReturnStatus::failure;
    }

    rmx_stats_init_message(&m_stats_message);
    return ReturnStatus::success;
}

void StatisticsReader::operator()()
{
    std::cout << "Start StatisticsReader thread" << std::endl;
    set_current_thread_affinity(m_cpu_core_affinity);
    
    ReturnStatus rc = init_stats_reader();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Failed to initialize Rivermax statistics" << std::endl;
        return;
    }
    
    try {

        // Main cycle
        do {
            while (read_stats_msg() == ReturnStatus::success) {

                if (skip()) {
                    continue;
                }
                handle_msg();
            }
            std::this_thread::sleep_for(std::chrono::seconds(m_update_time_sec));
        }
        while (likely(SignalHandler::get_received_signal() < 0)) ;
    }
    catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return;
    }
}

ReturnStatus StatisticsReader::read_stats_msg()
{
    auto status = rmx_stats_consumer_pop_message(&m_stats_consumer, &m_stats_message);
    if (likely(status == RMX_OK)) {
        return ReturnStatus::success;
    }
    if (SignalHandler::get_received_signal() != 0) {
        return ReturnStatus::signal_received;
    }
    return ReturnStatus::no_free_chunks;
}

bool StatisticsReader::skip() {
    uint32_t message_session_id = rmx_stats_get_session_id(&m_stats_message);
    uint32_t message_thread_id = rmx_stats_get_thread_id(&m_stats_message);

    if (((m_session_id != UINT_MAX) && (m_session_id != message_session_id)) ||
        ((m_thread_id != RMX_STATS_INVALID_THREAD_ID) && (m_thread_id != message_thread_id))) {
        return true;
    }
    return false;
}

void StatisticsReader::handle_msg()
{
    rmx_stats_type type = rmx_stats_get_type(&m_stats_message);

    auto it = m_statistics_handler_cb_map.find(type);
    if (it != m_statistics_handler_cb_map.end()) {
        it->second(this);
    } else {
        handle_unknown();
    }
}

void StatisticsReader::display_hdr()
{
    auto type = rmx_stats_get_type(&m_stats_message);
    const char* name = "unknown";
    stats_type_name_map_t::const_iterator it;

    it = stats_type_to_name.find(type);
    if (it != stats_type_to_name.end()) {
        name = it->second;
    }

    printf("%s\tSession id: %d",name, rmx_stats_get_session_id(&m_stats_message));
}

void StatisticsReader::handle_session_start()
{
    const rmx_stats_session_start_handle *handle = rmx_stats_get_session_start_handle(&m_stats_message);
    if (handle == nullptr) {
        std::cerr << "Invalid statistics data handle." << std::endl;
        return;
    }

    display_hdr();
    printf("\tHeight: %d\tWidth: %d\tFps: %.2f\n",
       rmx_stats_get_start_video_height(handle), rmx_stats_get_start_video_width(handle),
       rmx_stats_get_start_video_frames_per_second(handle));
}

void StatisticsReader::handle_session_stop()
{
    const rmx_stats_session_stop_handle *handle = rmx_stats_get_session_stop_handle(&m_stats_message);
    if (handle == nullptr) {
        std::cerr << "Invalid statistics data handle." << std::endl;
        return;
    }

    display_hdr();
    printf("Packets: %zu\tBytes: %zu\tStatus: %d\n", rmx_stats_get_stop_transfered_packets(handle),
           rmx_stats_get_stop_transfered_bytes(handle), rmx_stats_get_stop_status(handle));
}

void StatisticsReader::handle_session_run()
{
    const rmx_stats_session_runtime_handle *handle = rmx_stats_get_session_runtime_handle(&m_stats_message);
    if (handle == nullptr) {
        std::cerr << "Invalid statistics data handle." << std::endl;
        return;
    }

    display_hdr();
    printf("\tChunks: %zu\tStrides: %zu\tUser chunks: %d\tFree chunks: %d\tBusy chunks: %d\n",
           rmx_stats_get_runtime_committed_chunks(handle), rmx_stats_get_runtime_committed_strides(handle),
           rmx_stats_get_runtime_user_chunks(handle), rmx_stats_get_runtime_free_chunks(handle),
           rmx_stats_get_runtime_busy_chunks(handle));
}

void StatisticsReader::handle_tx_queue()
{
    const rmx_stats_tx_queue_handle *handle = rmx_stats_get_tx_queue_handle(&m_stats_message);
    if (handle == nullptr) {
        std::cerr << "Invalid statistics data handle." << std::endl;
        return;
    }

    display_hdr();
    printf("\tPackets: %zu\tBytes: %zu\tPacket wqes: %zu\tDummy wqes: %zu\tFree wqes: %u\n",
           rmx_stats_get_tx_queue_num_packets(handle),
           rmx_stats_get_tx_queue_num_bytes(handle), rmx_stats_get_tx_queue_packet_wqes(handle),
           rmx_stats_get_tx_queue_dummy_wqes(handle), rmx_stats_get_tx_queue_free_wqes(handle));
}

void StatisticsReader::handle_rx_queue()
{
    const rmx_stats_rx_queue_handle * handle = rmx_stats_get_rx_queue_handle(&m_stats_message);
    if (handle == nullptr) {
        std::cerr << "Invalid statistics data handle." << std::endl;
        return;
    }
    display_hdr();
    printf("\tPackets: %zu\tBytes: %zu\tStrides: %u\twqe: %u\tcrc: %zu\n",
           rmx_stats_get_rx_queue_num_packets(handle) ,
           rmx_stats_get_rx_queue_num_bytes(handle), rmx_stats_get_rx_queue_used_strides(handle),
           rmx_stats_get_rx_queue_wqe_strides(handle), rmx_stats_get_rx_queue_crc_errors(handle));
}

void StatisticsReader::handle_unknown()
{
    printf("unknown type 0x%x, session %d\n",
           rmx_stats_get_type(&m_stats_message), rmx_stats_get_session_id(&m_stats_message));
}

StatisticsReader::~StatisticsReader()
{
    if (m_stats_consumer_created) {
        rmx_stats_destroy_consumer(&m_stats_consumer);
    }
}

