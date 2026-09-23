/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <rivermax_api.h>
#include <sstream>
#include <utility>
#include <vector>

#include "rdk/services/cli/cli_manager.h"
#include "rdk/services/cli/options.h"
#include "rdk/services/cli/validators.h"
#include "rdk/services/cpu/cpu.h"
#include "rdk/services/media/media.h"
#include "rdk/services/sdp/sdp_defs.h"
#include "rdk/services/utils/defs.h"

using namespace rdk::services;

const char* CLIOptStr::LOCAL_IP = "-l,--local-ip";
const char* CLIOptStr::LOCAL_IPS = "--li,--local-ips";
const char* CLIOptStr::SRC_IP = "-s,--src-ip";
const char* CLIOptStr::SRC_IPS = "--si,--src-ips";
const char* CLIOptStr::DST_IP = "-d,--dst-ip";
const char* CLIOptStr::DST_IPS = "--di,--dst-ips";
const char* CLIOptStr::DST_PORT = "-p,--dst-port";
const char* CLIOptStr::DST_PORTS = "--dp,--dst-ports";
const char* CLIOptStr::THREADS = "-T,--threads";
const char* CLIOptStr::FLOWS = "-F,--flows";
const char* CLIOptStr::STREAMS = "-S,--streams";
const char* CLIOptStr::VERBOSE = "-v,--verbose";
const char* CLIOptStr::CHUNKS = "-C,--chunks";
const char* CLIOptStr::PACKETS = "-K,--packets";
const char* CLIOptStr::PAYLOAD_SIZE = "-y,--payload-size";
const char* CLIOptStr::APP_HDR_SIZE = "-e,--app-hdr-size";
const char* CLIOptStr::SLEEP_US = "-z,--sleep-us";
const char* CLIOptStr::SLEEP = "-z,--sleep";
const char* CLIOptStr::RATE_BPS = "-r,--rate-bps";
const char* CLIOptStr::RATE_BURST = "-b,--rate-burst";
const char* CLIOptStr::INTERNAL_CORE = "-i,--internal-core";
const char* CLIOptStr::APPLICATION_CORE = "-a,--application-core";
const char* CLIOptStr::CHECKSUM_HEADER = "-x,--checksum-header";
const char* CLIOptStr::WAIT_RETRY = "-w,--wait-retry";
const char* CLIOptStr::GPU_ID = "-g,--gpu-id";
const char* CLIOptStr::LOCK_GPU_CLOCKS = "-L,--lock-gpu-clocks";
const char* CLIOptStr::ALLOCATOR_TYPE = "-A,--allocator-type";
const char* CLIOptStr::REGISTER_MEMORY = "-M,--register-memory";
const char* CLIOptStr::HEADER_DATA_SPLIT = "-r,--header-data-split";
const char* CLIOptStr::APP_MEMORY_ALLOC = "-m,--app-memory-alloc";
const char* CLIOptStr::ENABLE_STATS_READER = "--esr,--enable-stats-reader";
const char* CLIOptStr::STATS_CORE = "-R,--statistics-core";
const char* CLIOptStr::STATS_SESSION_ID = "-P,--session-id-stats";
const char* CLIOptStr::STATS_REPORT_INTERVAL = "-I,--stats-interval";
const char* CLIOptStr::RX_STREAM_TYPE = "-R,--rx-stream-type";
const char* CLIOptStr::VIDEO_RESOLUTION = "--vr,--video-resolution";
const char* CLIOptStr::VIDEO_FRAME_RATE = "--vfr,--video-frame-rate";
const char* CLIOptStr::VIDEO_SAMPLING = "--vs,--video-sampling";
const char* CLIOptStr::VIDEO_SCAN_TYPE = "--vst,--video-scan-type";
const char* CLIOptStr::VIDEO_BIT_DEPTH = "--vbd,--video-bit-depth";
const char* CLIOptStr::ALPHA_BIT_DEPTH = "--abd,--alpha-bit-depth";
const char* CLIOptStr::ENABLE_VIDEO = "--enable-video";
const char* CLIOptStr::ENABLE_ALPHA = "--enable-alpha";
const char* CLIOptStr::ENABLE_EXTENDED_VIDEO_FORMATS = "--evf,--enable-extended-video-formats";
const char* CLIOptStr::ENABLE_AUDIO = "--enable-audio";
const char* CLIOptStr::ENABLE_ANCILLARY = "--enable-ancillary";
const char* CLIOptStr::VIDEO_FILE = "--vf,--video-file";
const char* CLIOptStr::AUDIO_FILE = "--af,--audio-file";
const char* CLIOptStr::ANCILLARY_FILE = "--anf,--ancillary-file";
const char* CLIOptStr::DYNAMIC_FILE_LOADING = "--dfl,--dynamic-file-loading";
const char* CLIOptStr::PTIME_US = "--apu,--audio-ptime-us";
const char* CLIOptStr::AUDIO_SAMPLING_RATE = "--asr,--audio-sampling-rate";
const char* CLIOptStr::AUDIO_ENCODING = "--ae,--audio-encoding";
const char* CLIOptStr::ANCILLARY_TYPES = "--ant,--ancillary-types";
const char* CLIOptStr::ANCILLARY_DATA_WORDS_COUNT = "--adwc,--ancillary-data-words-count";
const char* CLIOptStr::ENABLE_REDUNDANCY = "--er,--enable-redundancy";
const char* CLIOptStr::ALLOCATE_EXTRA_STREAM_BUFFERS = "--eb,--ext-buffers";

const char* CLIGroupStr::VIDEO_FORMAT_OPTIONS = "Video format options";
const char* CLIGroupStr::AUDIO_FORMAT_OPTIONS = "Audio format options";
const char* CLIGroupStr::ANCILLARY_FORMAT_OPTIONS = "Ancillary format options";

/**
 * @note: TODO: GPU related options are handled awkwardly and should be improved.
 *        The GPU allocator is now used implicitly rather than explicitly, which makes
 *        the interface unclear and should be refactored for better clarity.
 */
static const std::map<std::string, AllocatorTypeUI> UI_ALLOCATOR_TYPES{
    { "auto",           AllocatorTypeUI::Auto },
    { "malloc",         AllocatorTypeUI::Malloc },
    { "hugepage",       AllocatorTypeUI::HugePageDefault },
    { "hugepage-2m",    AllocatorTypeUI::HugePage2MB },
    { "hugepage-512m",  AllocatorTypeUI::HugePage512MB },
    { "hugepage-1g",    AllocatorTypeUI::HugePage1GB },
#ifdef CUDA_ENABLED
    { "gpuhostpinned",     AllocatorTypeUI::GPUHostPinned },
#endif
};

/**
 * @brief: RX stream types mapping to string.
 *
 * Maps string representations to @ref rmx_input_stream_params_type enum values.
 */
static const std::map<std::string, rmx_input_stream_params_type> UI_RX_STREAM_TYPES{
    { "raw",          RMX_INPUT_RAW_PACKET },
    { "app-protocol", RMX_INPUT_APP_PROTOCOL_PACKET }
};

/**
 * @brief: Ancillary data types mapping to string.
 *
 * Maps string representations to @ref AncillaryDataIdentifier.
 */
static const std::map<std::string, AncillaryDataIdentifier> UI_ANCILLARY_TYPES{
    { "timecode", ANCILLARY_TIMECODE_IDENTIFIER },
    { "afd",      ANCILLARY_AFD_IDENTIFIER },
    { "cc",       ANCILLARY_CLOSED_CAPTION_IDENTIFIER }
};

/**
 * @brief: Create a string to Enum mapping vector.
 *
 * @param [in] enums: The enum values.
 *
 * @return: Vector of string to Enum mapping pairs.
 */
template<typename EnumType>
std::vector<std::pair<std::string, EnumType>> create_mapping_vector(const std::vector<EnumType>& enums)
{
    std::vector<std::pair<std::string, EnumType>> mapping;
    mapping.reserve(enums.size());
    for (auto val : enums) {
        mapping.emplace_back(enum_to_string(val), val);
    }
    return mapping;
}

/**
 * @brief: Help text listing the supported video resolutions.
 */
std::string supported_video_resolutions_help()
{
    std::ostringstream help;
    help << "Supported: ";
    for (size_t i = 0; i < SUPPORTED_VIDEO_RESOLUTIONS.size(); ++i) {
        help << (i ? ", " : "") << SUPPORTED_VIDEO_RESOLUTIONS[i];
    }
    return help.str();
}

/**
 * @brief: Help text listing the supported video bit depths.
 */
std::string supported_video_bit_depths_help()
{
    std::ostringstream help;
    help << "Supported: ";
    for (size_t i = 0; i < SUPPORTED_VIDEO_BIT_DEPTHS.size(); ++i) {
        help << (i ? ", " : "") << enum_to_string(SUPPORTED_VIDEO_BIT_DEPTHS[i]);
    }
    return help.str();
}

/**
 * @brief: Builds a CLI option factory lambda.
 *
 * Wraps the given option creation statements in a lambda that catches
 * @ref CLI::Error, reports the failure and returns nullptr.
 *
 * @param [in] option_name: The name of the option, used for error reporting.
 */
#define RDK_CLI_OPTION_BUILDER(option_name, ...) \
    [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings) -> CLI::Option* { \
        try { \
            __VA_ARGS__ \
        } catch (const CLI::Error& error) { \
            std::cerr << "Failed to add CLI option [" << option_name << "]; error: " \
                      << error.what() << std::endl; \
            return nullptr; \
        } \
    }

/**
 * @brief: CLI options factory map.
 *
 * The map creates CLI options based on a given key.
 * It maps from the key to a dedicated lambda function that creates the CLI option and adds it
 * to the parser object.
 *
 * Future CLI options, that are common enough, can be added to this map and used using
 * @ref CLIParserManager::add_option method.
 *
 * @note: Builders are wrapped by @c RDK_CLI_OPTION_BUILDER, which catches
 *        @ref CLI::Error and returns nullptr on failure.
 */
cli_opt_factory_map_t CLIParserManager::s_cli_opt_fuctory {
    {
        CLIOptStr::LOCAL_IP,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::LOCAL_IP,
            return parser->add_option(CLIOptStr::LOCAL_IP,
                                      app_settings->local_ip,
                                      "Local IP of the NIC")
                            ->check(CLI::ValidIPV4);
        )
    },
    {
        CLIOptStr::LOCAL_IPS,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::LOCAL_IPS,
            return parser->add_option(CLIOptStr::LOCAL_IPS,
                                      app_settings->local_ips,
                                      "Local IPs of the NICs (comma-separated)")
                            ->delimiter(',')
                            ->check(CLI::ValidIPV4);
        )
    },
    {
        CLIOptStr::SRC_IP,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::SRC_IP,
            return parser->add_option(CLIOptStr::SRC_IP,
                                      app_settings->source_ip,
                                      "Source IP address")
                            ->check(CLI::ValidIPV4);
        )
    },
    {
        CLIOptStr::SRC_IPS,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::SRC_IPS,
            return parser->add_option(CLIOptStr::SRC_IPS,
                                      app_settings->source_ips,
                                      "Source IP addresses (comma-separated)")
                            ->delimiter(',')
                            ->check(CLI::ValidIPV4);
        )
    },
    {
        CLIOptStr::DST_IP,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::DST_IP,
            return parser->add_option(CLIOptStr::DST_IP,
                                      app_settings->destination_ip,
                                      "Destination IP of the connection",
                                      true)
                            ->check(CLI::ValidIPV4);
        )
    },
    {
        CLIOptStr::DST_IPS,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::DST_IPS,
            return parser->add_option(CLIOptStr::DST_IPS,
                                      app_settings->destination_ips,
                                      "Destination IP addresses (comma-separated)")
                            ->delimiter(',')
                            ->check(CLI::ValidIPV4);
        )
    },
    {
        CLIOptStr::DST_PORT,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::DST_PORT,
            return parser->add_option(CLIOptStr::DST_PORT,
                                      app_settings->destination_port,
                                      "Destination port of the connection",
                                      true)->check(CLI::Range(MIN_PORT, MAX_PORT));
        )
    },
    {
        CLIOptStr::DST_PORTS,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::DST_PORTS,
            return parser->add_option(CLIOptStr::DST_PORTS,
                                      app_settings->destination_ports,
                                      "Destination ports of the connection (comma-separated)",
                                      true)->delimiter(',')
                                      ->check(CLI::Range(MIN_PORT, MAX_PORT));
        )
    },
    {
        CLIOptStr::THREADS,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::THREADS,
            return parser->add_option(CLIOptStr::THREADS,
                                      app_settings->num_of_threads,
                                      "Number of threads to use",
                                      true)->check(CLI::PositiveNumber);
        )
    },
    {
        CLIOptStr::FLOWS,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::FLOWS,
            return parser->add_option(CLIOptStr::FLOWS,
                                      app_settings->num_of_total_flows,
                                      "Number of total flows",
                                      true)->check(CLI::PositiveNumber)->check(
                                          CLI::Range(MIN_NUM_OF_TOTAL_FLOWS, MAX_NUM_OF_TOTAL_FLOWS))->check(
                                              FlowNumberValidator(app_settings->num_of_total_streams));
        )
    },
    {
        CLIOptStr::STREAMS,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::STREAMS,
            return parser->add_option(CLIOptStr::STREAMS,
                                      app_settings->num_of_total_streams,
                                      "Number of total streams",
                                      true)->check(CLI::PositiveNumber);
        )
    },
    {
        CLIOptStr::VERBOSE,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::VERBOSE,
            return parser->add_flag(CLIOptStr::VERBOSE,
                                    app_settings->print_parameters,
                                    "Print verbose info");
        )
    },
    {
        CLIOptStr::CHUNKS,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::CHUNKS,
            return parser->add_option(CLIOptStr::CHUNKS,
                                      app_settings->num_of_chunks,
                                      "Number of memory chunks",
                                      true)->check(CLI::PositiveNumber);
        )
    },
    {
        CLIOptStr::PACKETS,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::PACKETS,
            return parser->add_option(CLIOptStr::PACKETS,
                                      app_settings->num_of_packets_in_chunk,
                                      "Number of packets in chunk (default depends on resolution)",
                                      true)->check(CLI::PositiveNumber);
        )
    },
    {
        CLIOptStr::PAYLOAD_SIZE,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::PAYLOAD_SIZE,
            return parser->add_option(CLIOptStr::PAYLOAD_SIZE,
                                      app_settings->packet_payload_size,
                                      "Packet's payload size",
                                      true)->check(CLI::PositiveNumber);
        )
    },
    {
        CLIOptStr::APP_HDR_SIZE,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::APP_HDR_SIZE,
            return parser->add_option(CLIOptStr::APP_HDR_SIZE,
                                      app_settings->packet_app_header_size,
                                      "Packet's application header size",
                                      true)->check(CLI::PositiveNumber);
        )
    },
    {
        CLIOptStr::SLEEP_US,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::SLEEP_US,
            return parser->add_option(CLIOptStr::SLEEP_US,
                                      app_settings->sleep_between_operations_us,
                                      "Sleep duration in microsecond after processing all thread's streams",
                                      true)->check(CLI::PositiveNumber);
        )
    },
    {
        CLIOptStr::SLEEP,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::SLEEP,
            return parser->add_flag(CLIOptStr::SLEEP,
                                    app_settings->sleep_between_operations,
                                    "Don't block on send, do sleep");
        )
    },
    {
        CLIOptStr::RATE_BPS,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::RATE_BPS,
            return parser->add_option(CLIOptStr::RATE_BPS,
                                      app_settings->rate.bps,
                                      "Rate limit in bits per second per flow")->check(CLI::PositiveNumber);
        )
    },
    {
        CLIOptStr::RATE_BURST,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::RATE_BURST,
            return parser->add_option(CLIOptStr::RATE_BURST,
                                      app_settings->rate.max_burst_in_packets,
                                      "Maximum allowed burst size in number of packets per flow",
                                      true)->check(CLI::PositiveNumber);
        )
    },
    {
        CLIOptStr::INTERNAL_CORE,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::INTERNAL_CORE,
            return parser->add_option(CLIOptStr::INTERNAL_CORE,
                                      app_settings->internal_thread_core,
                                      "CPU core affinity for Rivermax internal thread",
                                      true)->check(CLI::Range(NO_CPU_AFFINITY, MAX_CPU_ID));
        )
    },
    {
        CLIOptStr::APPLICATION_CORE,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::APPLICATION_CORE,
            return parser->add_option(CLIOptStr::APPLICATION_CORE,
                                      app_settings->app_threads_cores,
                                      "CPU core affinities comma separated list for the application")
                                      ->delimiter(',')->check(CLI::Range(NO_CPU_AFFINITY, MAX_CPU_ID));
        )
    },
    {
        CLIOptStr::CHECKSUM_HEADER,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::CHECKSUM_HEADER,
            return parser->add_flag(CLIOptStr::CHECKSUM_HEADER,
                                    app_settings->use_checksum_header,
                                    "Use checksum header");
        )
    },
    {
        CLIOptStr::WAIT_RETRY,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::WAIT_RETRY,
            return parser->add_option(CLIOptStr::WAIT_RETRY,
                                      app_settings->hw_queue_full_sleep_us,
                                      "Time to sleep in microseconds between subsequent "
                                      "commit retries if the HW queue is full")->check(CLI::PositiveNumber);
        )
    },
    {
        CLIOptStr::GPU_ID,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::GPU_ID,
            return parser->add_option(CLIOptStr::GPU_ID,
                                      app_settings->gpu_id,
                                      "Enable GPU direct, value is GPU id")->check(CLI::Range(0, MAX_GPU_ID));
        )
    },
    {
        CLIOptStr::LOCK_GPU_CLOCKS,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::LOCK_GPU_CLOCKS,
            return parser->add_option(CLIOptStr::LOCK_GPU_CLOCKS,
                                    app_settings->lock_gpu_clocks,
                                    "Lock GPU clocks to their maximum frequency",
                                    true)->check(CLI::Range(0, 1));
        )
    },
    {
        CLIOptStr::ALLOCATOR_TYPE,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::ALLOCATOR_TYPE,
            return parser->add_option(CLIOptStr::ALLOCATOR_TYPE,
                                      app_settings->allocator_type,
                                      "Memory allocator type")
                                      ->transform(CLI::CheckedTransformer(UI_ALLOCATOR_TYPES));
        )
    },
    {
        CLIOptStr::REGISTER_MEMORY,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::REGISTER_MEMORY,
            return parser->add_flag(CLIOptStr::REGISTER_MEMORY,
                                    app_settings->register_memory,
                                    "Register memory on the application side for better performance");
        )
    },
    {
        CLIOptStr::HEADER_DATA_SPLIT,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::HEADER_DATA_SPLIT,
            return parser->add_flag(CLIOptStr::HEADER_DATA_SPLIT,
                                    app_settings->header_data_split,
                                    "Use separate memory for headers and payloads");
        )
    },
    {
        CLIOptStr::APP_MEMORY_ALLOC,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::APP_MEMORY_ALLOC,
            return parser->add_flag(CLIOptStr::APP_MEMORY_ALLOC,
                                    app_settings->app_memory_alloc,
                                    "I\\O Memory allocated by application");
        )
    },
    {
        CLIOptStr::ENABLE_STATS_READER,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::ENABLE_STATS_READER,
            return parser->add_flag(CLIOptStr::ENABLE_STATS_READER,
                                    app_settings->enable_statistics_reader,
                                    "Enable statistics reader");
        )
    },
    {
        CLIOptStr::ENABLE_REDUNDANCY,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::ENABLE_REDUNDANCY,
            return parser->add_flag(CLIOptStr::ENABLE_REDUNDANCY,
                                    app_settings->enable_redundancy,
                                    "Enable SMPTE 2022-7 redundancy");
        )
    },
    {
        CLIOptStr::STATS_CORE,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::STATS_CORE,
            return parser->add_option(CLIOptStr::STATS_CORE,
                                      app_settings->statistics_reader_core,
                                      "CPU core affinity for statistics reader thread")
                                      ->check(CLI::Range(NO_CPU_AFFINITY, MAX_CPU_ID));
        )
    },
    {
        CLIOptStr::STATS_SESSION_ID,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::STATS_SESSION_ID,
            return parser->add_option(CLIOptStr::STATS_SESSION_ID,
                                      app_settings->session_id_stats,
                                      "Present runtime statistics of the given session id")
                                      ->check(CLI::NonNegativeNumber);
        )
    },
    {
        CLIOptStr::STATS_REPORT_INTERVAL,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::STATS_REPORT_INTERVAL,
            return parser->add_option(CLIOptStr::STATS_REPORT_INTERVAL,
                                      app_settings->stats_report_interval_ms,
                                      "Display statistics interval, ms",
                                      true)->check(CLI::NonNegativeNumber);
        )
    },
    {
        CLIOptStr::RX_STREAM_TYPE,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::RX_STREAM_TYPE,
            return parser->add_option(CLIOptStr::RX_STREAM_TYPE,
                                      app_settings->rx_stream_type,
                                      "RX stream type (raw, app-protocol)")
                                      ->transform(CLI::CheckedTransformer(UI_RX_STREAM_TYPES, CLI::ignore_case))
                                      ->default_val(RMX_INPUT_APP_PROTOCOL_PACKET);
        )
    },
    {
        CLIOptStr::VIDEO_RESOLUTION,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::VIDEO_RESOLUTION,
            return parser->add_option(CLIOptStr::VIDEO_RESOLUTION,
                                      app_settings->media.resolution,
                                      "Video resolution in format <width>x<height>. " +
                                      supported_video_resolutions_help())
                                      ->check(VideoResolutionValidator())
                                      ->default_val(Resolution(_1080_WIDTH, _1080_HEIGHT));
        )
    },
    {
        CLIOptStr::VIDEO_FRAME_RATE,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::VIDEO_FRAME_RATE,
            return parser->add_option(CLIOptStr::VIDEO_FRAME_RATE,
                                      app_settings->media.frame_rate,
                                      "Video frame rate in format <integer> or <numerator>/<denominator>")
                                      ->check(VideoFrameRateValidator())
                                      ->check(CLI::IsMember(SUPPORTED_VIDEO_FRAME_RATES))
                                      ->default_val(60);
        )
    },
    {
        CLIOptStr::VIDEO_SAMPLING,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::VIDEO_SAMPLING,
            return parser->add_option(CLIOptStr::VIDEO_SAMPLING,
                                      app_settings->media.sampling_type,
                                      "Video sampling type")
                                      ->transform(CLI::CheckedTransformer(create_mapping_vector(
                                                  SUPPORTED_VIDEO_SAMPLING_TYPES), CLI::ignore_case))
                                      ->default_val(enum_to_string(VideoSampling::YCbCr_4_2_2));
        )
    },
    {
        CLIOptStr::VIDEO_SCAN_TYPE,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::VIDEO_SCAN_TYPE,
            return parser->add_option(CLIOptStr::VIDEO_SCAN_TYPE,
                                      app_settings->media.video_scan_type,
                                      "Video scan type")
                                      ->transform(CLI::CheckedTransformer(create_mapping_vector(
                                                  SUPPORTED_VIDEO_SCAN_TYPES), CLI::ignore_case))
                                      ->default_val(enum_to_string(VideoScanType::Progressive));
        )
    },
    {
        CLIOptStr::VIDEO_BIT_DEPTH,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::VIDEO_BIT_DEPTH,
            return parser->add_option(CLIOptStr::VIDEO_BIT_DEPTH,
                                      app_settings->media.color_bit_depth,
                                      "Video bit depth. " + supported_video_bit_depths_help())
                                      ->transform(CLI::CheckedTransformer(create_mapping_vector(
                                                  EXTENDED_VIDEO_BIT_DEPTHS), CLI::ignore_case).description(""))
                                      ->default_val(enum_to_string(VideoBitDepth::_10));
        )
    },
    {
        CLIOptStr::ALPHA_BIT_DEPTH,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::ALPHA_BIT_DEPTH,
            return parser->add_option(CLIOptStr::ALPHA_BIT_DEPTH,
                                      app_settings->media.alpha_bit_depth,
                                      "Alpha channel bit depth. " + supported_video_bit_depths_help())
                                      ->transform(CLI::CheckedTransformer(create_mapping_vector(
                                                  EXTENDED_VIDEO_BIT_DEPTHS), CLI::ignore_case).description(""));
        )
    },
    {
        CLIOptStr::ENABLE_VIDEO,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::ENABLE_VIDEO,
            return parser->add_flag(CLIOptStr::ENABLE_VIDEO,
                                    app_settings->media.enable_video,
                                    "Enable video essence");
        )
    },
    {
        CLIOptStr::ENABLE_ALPHA,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::ENABLE_ALPHA,
            return parser->add_flag(CLIOptStr::ENABLE_ALPHA,
                                    app_settings->media.enable_alpha,
                                    "Enable Alpha channel with default bit depth of main video");
        )
    },
    {
        CLIOptStr::ENABLE_EXTENDED_VIDEO_FORMATS,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::ENABLE_EXTENDED_VIDEO_FORMATS,
            return parser->add_flag(CLIOptStr::ENABLE_EXTENDED_VIDEO_FORMATS,
                                    app_settings->media.enable_extended_video_formats,
                                    "Enable extended video formats");
        )
    },
    {
        CLIOptStr::VIDEO_FILE,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::VIDEO_FILE,
            return parser->add_option(CLIOptStr::VIDEO_FILE,
                                      app_settings->video_file,
                                      "Video file to send")
                                      ->check(CLI::ExistingFile);
        )
    },
    {
        CLIOptStr::AUDIO_FILE,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::AUDIO_FILE,
            return parser->add_option(CLIOptStr::AUDIO_FILE,
                                      app_settings->audio_file,
                                      "Audio file to send")
                                      ->check(CLI::ExistingFile);
        )
    },
    {
        CLIOptStr::ANCILLARY_FILE,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::ANCILLARY_FILE,
            return parser->add_option(CLIOptStr::ANCILLARY_FILE,
                                      app_settings->ancillary_file,
                                      "Ancillary data file (e.g., .srt for closed captions)")
                                      ->check(CLI::ExistingFile);
        )
    },
    {
        CLIOptStr::DYNAMIC_FILE_LOADING,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::DYNAMIC_FILE_LOADING,
            return parser->add_flag(CLIOptStr::DYNAMIC_FILE_LOADING,
                                      app_settings->dynamic_media_file_load,
                                      "Enable dynamic file loading: Load media frames on the fly during transmission)");
        )
    },
    {
        CLIOptStr::ENABLE_AUDIO,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::ENABLE_AUDIO,
            return parser->add_flag(CLIOptStr::ENABLE_AUDIO,
                                    app_settings->media.enable_audio,
                                    "Enable audio essence");
        )
    },
    {
        CLIOptStr::ENABLE_ANCILLARY,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::ENABLE_ANCILLARY,
            return parser->add_flag(CLIOptStr::ENABLE_ANCILLARY,
                                    app_settings->media.enable_ancillary,
                                    "Enable ancillary essence")
                                    ->group(CLIGroupStr::ANCILLARY_FORMAT_OPTIONS);
        )
    },
    {
        CLIOptStr::PTIME_US,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::PTIME_US,
            return parser->add_option(CLIOptStr::PTIME_US,
                                    app_settings->media.ptime_us,
                                    "Audio packet time in microseconds")
                                    ->default_val(1000);
        )
    },
    {
        CLIOptStr::AUDIO_SAMPLING_RATE,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::AUDIO_SAMPLING_RATE,
            return parser->add_option(CLIOptStr::AUDIO_SAMPLING_RATE,
                                    app_settings->media.audio_sampling_rate,
                                    "Audio sampling rate")
                                    ->transform(CLI::CheckedTransformer(create_mapping_vector(
                                                SUPPORTED_AUDIO_SAMPLING_RATES), CLI::ignore_case))
                                    ->default_val(enum_to_string(AudioSamplingRate::_48000));
        )
    },
    {
        CLIOptStr::AUDIO_ENCODING,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::AUDIO_ENCODING,
            return parser->add_option(CLIOptStr::AUDIO_ENCODING,
                                    app_settings->media.audio_encoding,
                                    "Audio encoding")
                                    ->transform(CLI::CheckedTransformer(create_mapping_vector(
                                                SUPPORTED_AUDIO_ENCODINGS), CLI::ignore_case))
                                    ->default_val(enum_to_string(AudioEncoding::L24));
        )
    },
    {
        CLIOptStr::ANCILLARY_TYPES,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::ANCILLARY_TYPES,
            auto enable_anc_opt = parser->get_option(CLIOptStr::ENABLE_ANCILLARY);
            return parser->add_option(CLIOptStr::ANCILLARY_TYPES,
                                    app_settings->media.ancillary_data_identifiers,
                                    "Ancillary data types to enable (comma-separated: timecode,afd,cc). "
                                    "Default: timecode,afd")
                                    ->delimiter(',')
                                    ->transform(CLI::CheckedTransformer(UI_ANCILLARY_TYPES, CLI::ignore_case))
                                    ->needs(enable_anc_opt);
        )
    },
    {
        CLIOptStr::ANCILLARY_DATA_WORDS_COUNT,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::ANCILLARY_DATA_WORDS_COUNT,
            return parser->add_option(CLIOptStr::ANCILLARY_DATA_WORDS_COUNT,
                                    app_settings->media.ancillary_data_words_count,
                                    "Ancillary data words count");
        )
    },
    {
        CLIOptStr::ALLOCATE_EXTRA_STREAM_BUFFERS,
        RDK_CLI_OPTION_BUILDER(CLIOptStr::ALLOCATE_EXTRA_STREAM_BUFFERS,
            return parser->add_flag(CLIOptStr::ALLOCATE_EXTRA_STREAM_BUFFERS,
                                    app_settings->allocate_extra_stream_buffers,
                                    "Allocate extra stream buffers (auxiliary and internal data)");
        )
    },
};
#undef RDK_CLI_OPTION_BUILDER
