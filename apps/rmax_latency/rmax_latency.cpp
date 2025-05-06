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

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
#include <memory>

#include <rivermax_api.h>

#include "rt_threads.h"
#include "rmax_latency.h"
#include "api/rmax_apps_lib_api.h"
#include "apps/rmax_base_app.h"
#include "services/utils/clock.h"
#include "services/utils/rtp_video.h"

using namespace ral::lib::core;
using namespace ral::lib::services;
using namespace ral::io_node;
using namespace ral::apps::rmax_latency;

static const std::string DEFAULT_SEND_IP = "224.1.1.1";
static const uint16_t DEFAULT_SEND_PORT = 2000;
static const std::string DEFAULT_RECEIVE_IP = "224.1.1.2";
static const uint16_t DEFAULT_RECEIVE_PORT = 2000;
static const size_t DEFAULT_SLEEP_USEC = 10;
static const size_t DEFAULT_MEASURE_SEC = 10;

constexpr const char* MODE_PINGPONG = "pp";
constexpr const char* MODE_FRAME = "frame";
constexpr const char* MODE_MEDIA = "media";

static const std::unordered_map<std::string, LatencyMode> oper_modes {
    {MODE_PINGPONG, LatencyMode::PingPong},
    {MODE_FRAME, LatencyMode::Frame},
    {MODE_MEDIA, LatencyMode::Media},
};

static const std::vector<double> default_percentiles {0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.50, 0.75, 0.90, 0.99, 0.999, 0.9999, 0.99999};

/**
 * Application constants.
 */
static constexpr const char* APP_DESCRIPTION = "NVIDIA Rivermax latency measurement application";

static constexpr const char* APP_USAGE_MESSAGE =
R"EOF(The latency measurement tool implements the following measurement modes:
  Ping-Pong mode - sending a single packet from Client to Server and back, using Rivermax Tx Generic API.
    In this mode packet round-trip time is measured, as well as the following internal Rivermax latencies:
       - Between chunk commit time and the time received in the HW Tx completion
       - Between the time received in the HW Rx completion and the time the SW received the next chunk.
  Frame mode - sending an array of data (representing a video frame or segment)
               from Client to Server using Rivermax Tx Generic API with max speed, without packet pacing.
    In this mode the total transmission time is measured from commit time of the first chunk in frame by the Client SW
    to the time of receiving the last chunk in frame by the Server SW.
    In this mode the application also shows the following intervals:
    - Transmit start latency:         from the time of the first SW commit to the time provided by the first Tx completion
    - Sending time of entire frame:   from the time of the first SW commit to the time provided by the last Tx completion
    - Receiving time of entire frame: from the time provided by the first HW Rx completion to the time when the last chunk was received by the SW
    - Receive latency:                from the time provided by the last HW Rx completion to the time of the last SW chunk reception
    - Reply end-to-end latency:       from the time of the reply chunk commit by the server SW to its reception by the client SW
  Media mode - sending a SMPTE ST2110-20 video stream from Client to Server.
    In this mode the following delays are measured:
    - On Client side: between a packet HW transmission time and its scheduled packet consumption time
                      (TPRj as specified by the SMPTE ST2110-21 Gapped PRS model)
    - On Server side: between a packet HW reception time and its scheduled packet consumption time
                      (TPRj as specified by the SMPTE ST2110-21 Gapped PRS model)
For measurements involving HW timestamps, PTP-synchronized HW Real Time Clock must be enabled in the NIC (see Rivermax user manual).

The following parameters are available in all measurement modes:
   -l,--local-ip       local IP of the NIC
   -d,--dst-ip         send stream destination IP
   -p,--dst-port       send stream destination port
   -r,--receive-ip     receive stream destination IP
   -o,--receive-port   receive stream destination port
   --mode              measurement mode (pp, frame, media)
   -c,--client         operate as client, if not specified - operate as server
On the client side the measurement interval can be specified:
   -m,--measure        measurement interval, sec
Configurable parameters for ping-pong mode:
   -y,--payload-size   packet size
   -A,--allocator-type memory allocator type
   -z,--sleep-us       pause between requests (usec)
Configurable parameters for frame mode:
   -y,--payload-size   packet size
   -A,--allocator-type memory allocator type (defaults to GPU if GPUDirect is used)
   -g,--gpu-id         GPU id for GPUDirect
   -z,--sleep-us       pause between requests (usec)
   -C,--chunks         number of chunks in frame
   -K,--packets        number of packets in chunk
Configurable parameters for media mode:
   -A,--allocator-type memory allocator type (defaults to GPU if GPUDirect is used)
   -g,--gpu-id         GPU id for GPUDirect
   -x,--stream-type    video stream type (1080p50, 1080p60, 2160p50, 2160p60)
   -K,--packets        number of packets in chunk

Examples:
  1. Ping-Pong mode
  client: rmax_latency -l 1.2.3.4 -d 224.2.3.4 -p 2000 -r 224.2.3.5 -o 2000 -i1 -a2 -t -c -m10
  server: rmax_latency -l 1.2.3.5 -d 224.2.3.5 -p 2000 -r 224.2.3.4 -o 2000 -i1 -a2 -t
  2. Frame Latency mode
  client: rmax_latency -l 1.2.3.4 -d 224.2.3.4 -p 2000 -r 224.2.3.5 -o 2000 -i1 -a2 -t -c -m10 --mode frame -C100 -K20
  server: rmax_latency -l 1.2.3.5 -d 224.2.3.5 -p 2000 -r 224.2.3.4 -o 2000 -i1 -a2 -t --mode frame -C100 -K20
  3. Media Latency mode
  client: rmax_latency -l 1.2.3.4 -d 224.2.3.4 -p 2000 -r 224.2.3.5 -o 2000 -i1 -a2 -t -c -m10 --mode media -x 1080p60 -K4
  server: rmax_latency -l 1.2.3.5 -d 224.2.3.5 -p 2000 -r 224.2.3.4 -o 2000 -i1 -a2 -t --mode media -x 1080p60;
)EOF";

int main(int argc, const char* argv[])
{
    LatencyApp app(argc, argv);

    ReturnStatus rc = app.run();
    if (rc == ReturnStatus::failure) {
        std::cerr << "Rivermax Latency app failed to run" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

LatencyApp::LatencyApp(int argc, const char* argv[]) :
    RmaxBaseApp(APP_DESCRIPTION, APP_USAGE_MESSAGE),
    m_receive_port{DEFAULT_RECEIVE_PORT},
    m_client{false},
    m_latency_mode{LatencyMode::PingPong},
    m_tx_header_mreg{nullptr, 0, 0},
    m_is_tx_header_mreg_registered{false},
    m_tx_payload_mreg{nullptr, 0, 0},
    m_is_tx_payload_mreg_registered{false},
    m_rx_header_mreg{nullptr, 0, 0},
    m_rx_payload_mreg{nullptr, 0, 0},
    m_measure_interval_sec{DEFAULT_MEASURE_SEC},
    m_device_interface{0},
    m_packets_option{nullptr}
{
    m_obj_init_status = initialize(argc, argv);
}

void LatencyApp::initialize_common_default_app_settings()
{
    RmaxBaseApp::initialize_common_default_app_settings();
    m_app_settings->destination_ip = DEFAULT_SEND_IP;
    m_app_settings->destination_port = DEFAULT_SEND_PORT;
    m_receive_ip = DEFAULT_RECEIVE_IP;
    m_receive_port = DEFAULT_RECEIVE_PORT;
    m_client = false;
    m_disable_ts = false;
    m_app_settings->num_of_packets_in_chunk = DEFAULT_NUM_OF_PACKETS_IN_CHUNK;
    m_app_settings->num_of_packets_in_chunk_specified = false;
    m_app_settings->num_of_chunks = 1;
    m_app_settings->video_stream_type = VIDEO_2110_20_1080p50;
    m_app_settings->packet_app_header_size = 0;
}

void LatencyApp::add_cli_options()
{
    std::shared_ptr<CLI::App> parser = m_cli_parser_manager->get_parser();

    m_cli_parser_manager->add_option(CLIOptStr::LOCAL_IP);
    m_cli_parser_manager->add_option(CLIOptStr::DST_IP)->
            description("Send stream destination IP");
    m_cli_parser_manager->add_option(CLIOptStr::DST_PORT)->
            description("Send stream destination port");
    parser->add_option("-r,--receive-ip", m_receive_ip,
                       "Receive stream destination IP", true)->
                       check(CLI::ValidIPV4);
    parser->add_option("-o,--receive-port", m_receive_port,
                       "Receive stream destination port", true)->
                       check(CLI::Range(MIN_PORT, MAX_PORT));
    parser->add_option("-m,--measure", m_measure_interval_sec,
                       "Measurement interval, sec", true)->
                       check(CLI::PositiveNumber);
    parser->add_flag("-c,--client", m_client,
                     "Operate as client, if not specified - operate as server");
    parser->add_flag("--disable-ts", m_disable_ts,
                     "Disable processing of realtime completion timestamps");
    parser->add_flag("--disable-percent", m_disable_percentile,
                     "Disable percentile calculation");
    parser->add_option("-x,--stream-type", m_app_settings->video_stream_type,
                       "Type of a video stream", true)->
                       check(CLI::IsMember(SUPPORTED_STREAMS));
    parser->add_option("--mode", m_latency_mode,
                       "Latency measurement mode", true)->
                       transform(CLI::Transformer(oper_modes));
    m_cli_parser_manager->add_option(CLIOptStr::PAYLOAD_SIZE);
    m_cli_parser_manager->add_option(CLIOptStr::CHUNKS);
    m_packets_option = m_cli_parser_manager->add_option(CLIOptStr::PACKETS);
    m_cli_parser_manager->add_option(CLIOptStr::SLEEP_US)->
            description("Sleep duration in microsecond between latency measurement iterations");
    m_cli_parser_manager->add_option(CLIOptStr::INTERNAL_CORE);
    m_cli_parser_manager->add_option(CLIOptStr::APPLICATION_CORE);
    m_cli_parser_manager->add_option(CLIOptStr::ALLOCATOR_TYPE);
    m_cli_parser_manager->add_option(CLIOptStr::GPU_ID);
    m_cli_parser_manager->add_option(CLIOptStr::VERBOSE);
}

LatencyMode LatencyApp::get_latency_mode()
{
    return m_latency_mode;
}

ReturnStatus LatencyApp::run()
{
    if (m_obj_init_status != ReturnStatus::obj_init_success) {
        return m_obj_init_status;
    }

    if (m_client) {
        if (m_app_settings->destination_ip == "") {
            m_app_settings->destination_ip = DEFAULT_SEND_IP;
        }
        if (m_receive_ip == "") {
            m_receive_ip = DEFAULT_RECEIVE_IP;
        }
    } else {
        if (m_app_settings->destination_ip == "") {
            m_app_settings->destination_ip = DEFAULT_RECEIVE_IP;
        }
        if (m_receive_ip == "") {
            m_receive_ip = DEFAULT_SEND_IP;
        }
    }

    if (m_app_settings->destination_port == 0) {
        m_app_settings->destination_port = DEFAULT_SEND_PORT;
    }
    if (m_receive_port == 0) {
        m_receive_port = DEFAULT_RECEIVE_PORT;
    }

    if (m_app_settings->gpu_id != INVALID_GPU_ID) {
        if (((m_latency_mode == LatencyMode::Frame) || (m_latency_mode == LatencyMode::PingPong)) &&
            (m_app_settings->packet_payload_size <= RTP_HEADER_SIZE)) {
            std::cerr << "Packet length with GPU-Direct must be at least "
                      << RTP_HEADER_SIZE + 1 << " bytes" << std::endl;
            return ReturnStatus::failure;
        }
    }

    try {
        ReturnStatus rc = initialize_threads();
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

ReturnStatus LatencyApp::initialize_rivermax_resources()
{
    rt_set_realtime_class();

    return m_rmax_apps_lib.initialize_rivermax({m_app_settings->internal_thread_core}, true);
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
    if (m_disable_ts) {
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
    node_settings.receive_ip = m_receive_ip;
    node_settings.receive_port = m_receive_port;
    node_settings.client_mode = m_client;
    node_settings.measure_interval = m_measure_interval_sec;
    node_settings.track_completions = !m_disable_ts;
    node_settings.percentiles = m_disable_percentile ? std::vector<double>{} : default_percentiles;
    node_settings.gpu_direct_tx = m_client && gpu_direct_enabled();
    node_settings.gpu_direct_rx = !m_client && gpu_direct_enabled();

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
                    TwoTupleFlow(0, m_app_settings->local_ip, m_receive_port));
            break;
        case LatencyMode::Frame:
            m_io_node = std::unique_ptr<LatencyIONode>(
                    new FrameIONode(node_settings,
                                    m_header_allocator->get_memory_utils(),
                                    m_payload_allocator->get_memory_utils(),
                                    LatencyApp::get_time_ns));
            m_io_node->initialize_send_stream();
            m_io_node->initialize_receive_stream(
                    TwoTupleFlow(0, m_app_settings->local_ip, m_receive_port));
            break;
        case LatencyMode::Media:
            if (m_client) {
                if (m_packets_option->results().size() != 0) {
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
                    TwoTupleFlow(0, m_app_settings->local_ip, m_receive_port));
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
