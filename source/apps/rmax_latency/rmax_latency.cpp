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

#include <cassert>

#include "rdk/apps/rmax_latency/rmax_latency.h"
#include "rdk/services/utils/clock.h"
#include "rdk/services/media/media.h"

using namespace rivermax::dev_kit::apps::rmax_latency;

constexpr const char* MODE_PINGPONG = "pp";
constexpr const char* MODE_FRAME = "frame";
constexpr const char* MODE_MEDIA = "media";

static const std::unordered_map<std::string, LatencyMode> oper_modes {
    {MODE_PINGPONG, LatencyMode::PingPong},
    {MODE_FRAME, LatencyMode::Frame},
    {MODE_MEDIA, LatencyMode::Media},
};

static const std::vector<double> default_percentiles {0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.50, 0.75, 0.90, 0.99, 0.999, 0.9999, 0.99999};

void LatencySettings::init_default_values()
{
    AppSettings::init_default_values();
    destination_ip = DEFAULT_SEND_IP;
    destination_port = DEFAULT_SEND_PORT;
    receive_ip = DEFAULT_RECEIVE_IP;
    receive_port = DEFAULT_RECEIVE_PORT;
    client = false;
    disable_ts = false;
    disable_percentile = false;
    latency_mode = LatencyMode::PingPong;
    measure_interval_sec = DEFAULT_MEASURE_SEC;
    num_of_packets_in_chunk = DEFAULT_NUM_OF_PACKETS_IN_CHUNK;
    num_of_packets_in_chunk_specified = false;
    num_of_chunks = 1;
    packet_app_header_size = 0;
    app_memory_alloc = true;
    register_memory = true;
}

ReturnStatus LatencySettingsValidator::validate(const std::shared_ptr<LatencySettings>& settings) const
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
    rc = ValidatorUtils::validate_ip4_address(settings->receive_ip);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_ip4_port(settings->receive_port);
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
    if (settings->register_memory && !settings->app_memory_alloc) {
        std::cerr << "Register memory option is supported only with application memory allocation" << std::endl;
        return ReturnStatus::failure;
    }
#ifdef CUDA_ENABLED
    if (settings->gpu_id != INVALID_GPU_ID) {
        if (((settings->latency_mode == LatencyMode::Frame) || (settings->latency_mode == LatencyMode::PingPong)) &&
            (settings->packet_payload_size <= RTP_HEADER_SIZE)) {
            std::cerr << "Packet length with GPU-Direct must be at least " << RTP_HEADER_SIZE + 1 << " bytes" << std::endl;
            return ReturnStatus::failure;
        }
    }
#endif

    return ReturnStatus::success;
}

ReturnStatus LatencyCLISettingsBuilder::add_cli_options(std::shared_ptr<LatencySettings>& settings)
{
    if (m_cli_parser_manager == nullptr) {
        std::cerr << "CLI parser manager is not initialized" << std::endl;
        return ReturnStatus::failure;
    }
    m_cli_parser_manager->add_option(CLIOptStr::LOCAL_IP);
    m_cli_parser_manager->add_option(CLIOptStr::DST_IP)->
            description("Send stream destination IP");
    m_cli_parser_manager->add_option(CLIOptStr::DST_PORT)->
            description("Send stream destination port");
    auto parser = m_cli_parser_manager->get_parser();
    parser->add_option("-r,--receive-ip", settings->receive_ip,
                       "Receive stream destination IP", true)->
                       check(CLI::ValidIPV4);
    parser->add_option("-o,--receive-port", settings->receive_port,
                       "Receive stream destination port", true)->
                       check(CLI::Range(MIN_PORT, MAX_PORT));
    parser->add_option("-m,--measure", settings->measure_interval_sec,
                       "Measurement interval, sec", true)->
                       check(CLI::PositiveNumber);
    parser->add_flag("-c,--client", settings->client,
                     "Operate as client, if not specified - operate as server");
    parser->add_flag("--disable-ts", settings->disable_ts,
                     "Disable processing of realtime completion timestamps");
    parser->add_flag("--disable-percent", settings->disable_percentile,
                     "Disable percentile calculation");
    m_cli_parser_manager->add_option(CLIOptStr::VIDEO_RESOLUTION)
        ->group(CLIGroupStr::VIDEO_FORMAT_OPTIONS);
    m_cli_parser_manager->add_option(CLIOptStr::VIDEO_FRAME_RATE)
        ->group(CLIGroupStr::VIDEO_FORMAT_OPTIONS);
    parser->add_option("--mode", settings->latency_mode,
                       "Latency measurement mode", true)->
                       transform(CLI::Transformer(oper_modes));
    m_cli_parser_manager->add_option(CLIOptStr::PAYLOAD_SIZE);
    m_cli_parser_manager->add_option(CLIOptStr::CHUNKS);
    m_cli_parser_manager->add_option(CLIOptStr::PACKETS);
    m_cli_parser_manager->add_option(CLIOptStr::SLEEP_US)->
            description("Sleep duration in microsecond between latency measurement iterations");
    m_cli_parser_manager->add_option(CLIOptStr::INTERNAL_CORE);
    m_cli_parser_manager->add_option(CLIOptStr::APPLICATION_CORE);
    m_cli_parser_manager->add_option(CLIOptStr::ALLOCATOR_TYPE);
#ifdef CUDA_ENABLED
    m_cli_parser_manager->add_option(CLIOptStr::GPU_ID);
    m_cli_parser_manager->add_option(CLIOptStr::LOCK_GPU_CLOCKS);
#endif
    m_cli_parser_manager->add_option(CLIOptStr::VERBOSE);

    return ReturnStatus::success;
}

LatencyApp::LatencyApp(std::shared_ptr<ISettingsBuilder<LatencySettings>> settings_builder) :
    RmaxBaseApp(),
    m_settings_builder(std::move(settings_builder)),
    m_tx_header_mreg{nullptr, 0, 0},
    m_is_tx_header_mreg_registered{false},
    m_tx_payload_mreg{nullptr, 0, 0},
    m_is_tx_payload_mreg_registered{false},
    m_rx_header_mreg{nullptr, 0, 0},
    m_rx_payload_mreg{nullptr, 0, 0},
    m_device_interface{0}
{
}

ReturnStatus LatencyApp::initialize_app_settings()
{
    if (m_settings_builder == nullptr) {
        std::cerr << "Settings builder is not initialized" << std::endl;
        return ReturnStatus::failure;
    }
    m_latency_settings = std::make_shared<LatencySettings>();
    ReturnStatus rc = m_settings_builder->build(m_latency_settings);
    if (rc == ReturnStatus::success) {
        m_app_settings = m_latency_settings;
        return ReturnStatus::success;
    }
    if (rc != ReturnStatus::success_cli_help) {
        std::cerr << "Failed to build settings" << std::endl;
    }
    m_obj_init_status = rc;
    return rc;
}

ReturnStatus LatencyApp::initialize()
{
    ReturnStatus rc  = RmaxBaseApp::initialize();

    if (rc != ReturnStatus::obj_init_success) {
        return m_obj_init_status;
    }

    try {
        rc = initialize_threads();
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to initialize application threads" << std::endl;
            return ReturnStatus::failure;
        }
        rc = allocate_app_memory();
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to allocate the memory required for the application" << std::endl;
            return rc;
        }
        distribute_memory_for_streams();
    }
    catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return ReturnStatus::failure;
    }

    m_obj_init_status = ReturnStatus::obj_init_success;
    return m_obj_init_status;
}

LatencyMode LatencyApp::get_latency_mode()
{
    return m_latency_settings->latency_mode;
}

ReturnStatus LatencyApp::run()
{
    if (m_obj_init_status != ReturnStatus::obj_init_success) {
        return m_obj_init_status;
    }

    try {
        std::cout << *m_io_node;
        auto thread = std::thread(std::ref(*m_io_node));
        thread.join();
        unregister_app_memory();
    }
    catch (const std::exception& error) {
        std::cerr << "application error: " << error.what() << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus LatencyApp::initialize_connection_parameters()
{
    ReturnStatus rc = RmaxBaseApp::initialize_connection_parameters();
    if (rc != ReturnStatus::success) {
        return rc;
    }
    return init_app_device_iface(m_device_interface);
}

ReturnStatus LatencyApp::set_rivermax_clock()
{
    if (m_latency_settings->disable_ts) {
        std::cout << "Using default system clock" << std::endl;
        return ReturnStatus::success;
    }
    std::cout << "Switching to PTP clock" << std::endl;
    return set_rivermax_ptp_clock(&m_device_interface);
}

ReturnStatus LatencyApp::initialize_threads()
{
    LatencyNodeSettings node_settings;
    node_settings.app = m_app_settings;
    node_settings.receive_ip = m_latency_settings->receive_ip;
    node_settings.receive_port = m_latency_settings->receive_port;
    node_settings.client_mode = m_latency_settings->client;
    node_settings.measure_interval = m_latency_settings->measure_interval_sec;
    node_settings.track_completions = !m_latency_settings->disable_ts;
    node_settings.percentiles = m_latency_settings->disable_percentile ? std::vector<double>{} : default_percentiles;
    node_settings.gpu_direct_tx = m_latency_settings->client && gpu_direct_enabled();
    node_settings.gpu_direct_rx = !m_latency_settings->client && gpu_direct_enabled();

    switch (get_latency_mode()) {
        case LatencyMode::PingPong:
            node_settings.gpu_direct_tx = gpu_direct_enabled();
            node_settings.gpu_direct_rx = gpu_direct_enabled();
            m_io_node = std::unique_ptr<LatencyIONode>(
                    new PingPongIONode(node_settings,
                                       m_header_allocator->get_memory_utils(),
                                       m_payload_allocator->get_memory_utils(),
                                       LatencyApp::get_time_ns));
            m_io_node->initialize_send_stream();
            m_io_node->initialize_receive_stream(
                    TwoTupleFlow(0, m_app_settings->local_ip, m_latency_settings->receive_port));
            break;
        case LatencyMode::Frame:
            m_io_node = std::unique_ptr<LatencyIONode>(
                    new FrameIONode(node_settings,
                                    m_header_allocator->get_memory_utils(),
                                    m_payload_allocator->get_memory_utils(),
                                    LatencyApp::get_time_ns));
            m_io_node->initialize_send_stream();
            m_io_node->initialize_receive_stream(
                    TwoTupleFlow(0, m_app_settings->local_ip, m_latency_settings->receive_port));
            break;
        case LatencyMode::Media:
            if (m_latency_settings->client) {
                if (m_app_settings->num_of_packets_in_chunk != LatencySettings::DEFAULT_NUM_OF_PACKETS_IN_CHUNK) {
                    m_app_settings->num_of_packets_in_chunk_specified = true;
                }
                m_io_node = std::unique_ptr<LatencyIONode>(
                        new MediaTxIONode(node_settings,
                                          m_header_allocator->get_memory_utils(),
                                          m_payload_allocator->get_memory_utils(),
                                          LatencyApp::get_time_ns));
            } else {
                m_io_node = std::unique_ptr<LatencyIONode>(
                        new MediaRxIONode(node_settings,
                                          m_header_allocator->get_memory_utils(),
                                          m_payload_allocator->get_memory_utils(),
                                          LatencyApp::get_time_ns));
            }
            m_io_node->initialize_send_stream();
            m_io_node->initialize_receive_stream(
                    TwoTupleFlow(0, m_app_settings->local_ip, m_latency_settings->receive_port));
            break;
    }
    return ReturnStatus::success;
}

void* LatencyApp::allocate_and_align_header(size_t size)
{
    return m_header_allocator->allocate_aligned(size, m_header_allocator->get_page_size());
}

void* LatencyApp::allocate_and_align_payload(size_t size)
{
    size = m_payload_allocator->align_length(size);
    return m_payload_allocator->allocate_aligned(size, m_payload_allocator->get_page_size());
}

ReturnStatus LatencyApp::init_app_device_iface(rmx_device_iface& device_iface)
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

ReturnStatus LatencyApp::allocate_app_memory()
{
    size_t tx_header_size;
    size_t tx_payload_size;
    size_t rx_header_size;
    size_t rx_payload_size;

    if (m_io_node->query_memory_size(tx_header_size, tx_payload_size,
                                     rx_header_size, rx_payload_size) != ReturnStatus::success) {
        std::cout << "Error detecting stream memory requirements";
        return ReturnStatus::failure;
    }

    std::cout << "Application requires " << tx_payload_size << " bytes of memory for Tx and ";
    std::cout << tx_header_size << " bytes for Tx separate headers ";
    std::cout << rx_payload_size << " bytes for Rx and ";
    std::cout << rx_header_size << " bytes for Rx separate headers" << std::endl;

    rmx_mem_reg_params mem_registry;
    rmx_init_mem_registry(&mem_registry, &m_device_interface);

    if (tx_header_size) {
        m_tx_header_mreg.addr = allocate_and_align_header(tx_header_size);
        m_tx_header_mreg.length = tx_header_size;
        m_tx_header_mreg.mkey = 0;

        if (!m_tx_header_mreg.addr) {
            std::cerr << "Failed to allocate application Tx header memory" << std::endl;
            return ReturnStatus::failure;
        }

        std::cout << "Allocated for Tx headers " << m_tx_header_mreg.length <<
            " bytes at address " << static_cast<void *>(m_tx_header_mreg.addr) << std::endl;

        rmx_status status = rmx_register_memory(&m_tx_header_mreg, &mem_registry);
        if (status != RMX_OK) {
            std::cerr << "Failed to register Tx header memory with status: " << status << std::endl;
            return ReturnStatus::failure;
        }
        m_is_tx_header_mreg_registered = true;
    }

    if (tx_payload_size) {
        /* if HDS for Tx stream is on, allocate payload in a separate storage */
        if (tx_header_size) {
            m_tx_payload_mreg.addr = allocate_and_align_payload(tx_payload_size);
        } else {
            m_tx_payload_mreg.addr = allocate_and_align_header(tx_payload_size);
        }
        m_tx_payload_mreg.length = tx_payload_size;
        m_tx_payload_mreg.mkey = 0;

        if (!m_tx_payload_mreg.addr) {
            std::cerr << "Failed to allocate application Tx payload memory" << std::endl;
            return ReturnStatus::failure;
        }

        std::cout << "Allocated for Tx payload " << m_tx_payload_mreg.length <<
            " bytes at address " << static_cast<void *>(m_tx_payload_mreg.addr) << std::endl;

        rmx_status status = rmx_register_memory(&m_tx_payload_mreg, &mem_registry);
        if (status != RMX_OK) {
            std::cerr << "Failed to register Tx payload memory with status: " << status << std::endl;
            return ReturnStatus::failure;
        }
        m_is_tx_payload_mreg_registered = true;
    }

    if (rx_header_size) {
        m_rx_header_mreg.addr = allocate_and_align_header(rx_header_size);
        m_rx_header_mreg.length = rx_header_size;
        m_rx_header_mreg.mkey = 0;

        if (!m_rx_header_mreg.addr) {
            std::cerr << "Failed to allocate application Rx header memory" << std::endl;
            return ReturnStatus::failure;
        }

        std::cout << "Allocated for Rx headers " << m_rx_header_mreg.length <<
            " bytes at address " << static_cast<void *>(m_rx_header_mreg.addr) << std::endl;
    }

    if (rx_payload_size) {
        /* if HDS for Rx stream is on, allocate payload in a separate storage */
        if (rx_header_size) {
            m_rx_payload_mreg.addr = allocate_and_align_payload(rx_payload_size);
        } else {
            m_rx_payload_mreg.addr = allocate_and_align_header(rx_payload_size);
        }
        m_rx_payload_mreg.length = rx_payload_size;
        m_rx_payload_mreg.mkey = 0;

        if (!m_rx_payload_mreg.addr) {
            std::cerr << "Failed to allocate application Rx payload memory" << std::endl;
            return ReturnStatus::failure;
        }

        std::cout << "Allocated for Rx payload " << m_rx_payload_mreg.length <<
            " bytes at address " << static_cast<void *>(m_rx_payload_mreg.addr) << std::endl;
    }

    return ReturnStatus::success;
}

void LatencyApp::unregister_app_memory()
{
    if (m_is_tx_header_mreg_registered) {
        rmx_status status = rmx_deregister_memory(&m_tx_header_mreg, &m_device_interface);
        if (status != RMX_OK) {
            std::cerr << "Failed to de-register Tx header with status: "
                << status << std::endl;
        }
    }

    if (m_is_tx_payload_mreg_registered) {
        rmx_status status = rmx_deregister_memory(&m_tx_payload_mreg, &m_device_interface);
        if (status != RMX_OK) {
            std::cerr << "Failed to de-register Tx payload memory with status: "
                << status << std::endl;
        }
    }
}

void LatencyApp::distribute_memory_for_streams()
{
    m_io_node->distribute_memory_for_streams(m_tx_header_mreg, m_tx_payload_mreg,
                                             m_rx_header_mreg, m_rx_payload_mreg);
}

uint64_t LatencyApp::get_time_ns(void* context)
{
    NOT_IN_USE(context);
    uint64_t time_ns;
    if (rmx_get_time(RMX_TIME_PTP, &time_ns) != RMX_OK) {
        return 0;
    }
    return time_ns;
}
