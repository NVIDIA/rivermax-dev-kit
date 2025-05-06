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

#include <memory>
#include <cstring>
#include <map>
#include <vector>

#include "CLI/CLI.hpp"
#include "rt_threads.h"

#include "rdk/services/sdp/sdp_defs.h"
#include "rdk/services/utils/defs.h"
#include "rdk/services/media/media.h"
#include "rdk/services/cli/options.h"
#include "rdk/services/cli/cli_manager.h"
#include "rdk/services/cli/validators.h"

using namespace rivermax::dev_kit::services;

const char* CLIOptStr::LOCAL_IP = "-l,--local-ip";
const char* CLIOptStr::LOCAL_IPS = "-l,--local-ips";
const char* CLIOptStr::SRC_IP = "-s,--src-ip";
const char* CLIOptStr::SRC_IPS = "-s,--src-ips";
const char* CLIOptStr::DST_IP = "-d,--dst-ip";
const char* CLIOptStr::DST_IPS = "-d,--dst-ips";
const char* CLIOptStr::DST_PORT = "-p,--dst-port";
const char* CLIOptStr::DST_PORTS = "-p,--dst-ports";
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
const char* CLIOptStr::VIDEO_RESOLUTION = "--vr,--video-resolution";
const char* CLIOptStr::VIDEO_FRAME_RATE = "--vfr,--video-frame-rate";
const char* CLIOptStr::VIDEO_SAMPLING = "--vs,--video-sampling";
const char* CLIOptStr::VIDEO_BIT_DEPTH = "--vbd,--video-bit-depth";
const char* CLIOptStr::VIDEO_FILE = "--vf,--video-file";
const char* CLIOptStr::DYNAMIC_FILE_LOADING = "--dfl,--dynamic-file-loading";

const char* CLIGroupStr::VIDEO_FORMAT_OPTIONS = "Video format options";

static const std::map<std::string, AllocatorTypeUI> UI_ALLOCATOR_TYPES{
    { "auto",           AllocatorTypeUI::Auto },
    { "malloc",         AllocatorTypeUI::Malloc },
    { "hugepage",       AllocatorTypeUI::HugePageDefault },
    { "hugepage-2m",    AllocatorTypeUI::HugePage2MB },
    { "hugepage-512m",  AllocatorTypeUI::HugePage512MB },
    { "hugepage-1g",    AllocatorTypeUI::HugePage1GB },
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
 * @brief: CLI options factory map.
 *
 * The map creates CLI options based on a given key.
 * It maps from the key to a dedicated lambda function that creates the CLI option and adds it
 * to the parser object.
 *
 * Future CLI options, that are common enough, can be added to this map and used using
 * @ref CLIParserManager::add_option method.
 */
cli_opt_factory_map_t CLIParserManager::s_cli_opt_fuctory {
    {
        CLIOptStr::LOCAL_IP,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::LOCAL_IP,
                                      app_settings->local_ip,
                                      "Local IP of the NIC")->check(CLI::ValidIPV4)->required();
        }
    },
    {
        CLIOptStr::LOCAL_IPS,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::LOCAL_IPS,
                                      app_settings->local_ips,
                                      "Local IPs of the NICs (comma-separated)")
                            ->delimiter(',')
                            ->check(CLI::ValidIPV4)
                            ->required();
        }
    },
    {
        CLIOptStr::SRC_IP,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::SRC_IP,
                                      app_settings->source_ip,
                                      "Source IP address")
                            ->check(CLI::ValidIPV4)->required();
        }
    },
    {
        CLIOptStr::SRC_IPS,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::SRC_IPS,
                                      app_settings->source_ips,
                                      "Source IP addresses (comma-separated)")
                            ->delimiter(',')
                            ->check(CLI::ValidIPV4)
                            ->required();
        }
    },
    {
        CLIOptStr::DST_IP,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::DST_IP,
                                      app_settings->destination_ip,
                                      "Destination IP of the connection",
                                      true)->check(CLI::ValidIPV4);
        }
    },
    {
        CLIOptStr::DST_IPS,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::DST_IPS,
                                      app_settings->destination_ips,
                                      "Destination IP addresses (comma-separated)")
                            ->delimiter(',')
                            ->check(CLI::ValidIPV4)
                            ->required();
        }
    },
    {
        CLIOptStr::DST_PORT,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::DST_PORT,
                                      app_settings->destination_port,
                                      "Destination port of the connection",
                                      true)->check(CLI::Range(MIN_PORT, MAX_PORT));
        }
    },
    {
        CLIOptStr::DST_PORTS,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::DST_PORTS,
                                      app_settings->destination_ports,
                                      "Destination ports of the connection (comma-separated)",
                                      true)
                            ->delimiter(',')
                            ->check(CLI::Range(MIN_PORT, MAX_PORT));
        }
    },
    {
        CLIOptStr::THREADS,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::THREADS,
                                      app_settings->num_of_threads,
                                      "Number of threads to use",
                                      true)->check(CLI::PositiveNumber);
        }
    },
    {
        CLIOptStr::FLOWS,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::FLOWS,
                                      app_settings->num_of_total_flows,
                                      "Number of total flows",
                                      true)->check(CLI::PositiveNumber)->check(
                                          CLI::Range(MIN_NUM_OF_TOTAL_FLOWS, MAX_NUM_OF_TOTAL_FLOWS))->check(
                                              FlowNumberValidator(app_settings->num_of_total_streams));
        }
    },
    {
        CLIOptStr::STREAMS,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            auto option = \
                parser->add_option(CLIOptStr::STREAMS,
                                   app_settings->num_of_total_streams,
                                   "Number of total streams",
                                    true)->check(CLI::PositiveNumber);
            return option;
        }
    },
    {
        CLIOptStr::VERBOSE,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_flag(CLIOptStr::VERBOSE,
                                    app_settings->print_parameters,
                                    "Print verbose info");
        }
    },
    {
        CLIOptStr::CHUNKS,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::CHUNKS,
                                      app_settings->num_of_chunks,
                                      "Number of memory chunks",
                                      true)->check(CLI::PositiveNumber);
        }
    },
    {
        CLIOptStr::PACKETS,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::PACKETS,
                                      app_settings->num_of_packets_in_chunk,
                                      "Number of packets in chunk",
                                      true)->check(CLI::PositiveNumber);
        }
    },
    {
        CLIOptStr::PAYLOAD_SIZE,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::PAYLOAD_SIZE,
                                      app_settings->packet_payload_size,
                                      "Packet's payload size",
                                      true)->check(CLI::PositiveNumber);
        }
    },
    {
        CLIOptStr::APP_HDR_SIZE,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::APP_HDR_SIZE,
                                      app_settings->packet_app_header_size,
                                      "Packet's application header size",
                                      true)->check(CLI::PositiveNumber);
        }
    },
    {
        CLIOptStr::SLEEP_US,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::SLEEP_US,
                                      app_settings->sleep_between_operations_us,
                                      "Sleep duration in microsecond after processing all thread's streams",
                                      true)->check(CLI::PositiveNumber);
        }
    },
    {
        CLIOptStr::SLEEP,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_flag(CLIOptStr::SLEEP,
                                    app_settings->sleep_between_operations,
                                    "Don't block on send, do sleep");
        }
    },
    {
        CLIOptStr::RATE_BPS,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::RATE_BPS,
                                      app_settings->rate.bps,
                                      "Rate limit in bits per second per flow")->check(CLI::PositiveNumber);
        }
    },
    {
        CLIOptStr::RATE_BURST,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::RATE_BURST,
                                      app_settings->rate.max_burst_in_packets,
                                      "Maximum allowed burst size in number of packets per flow",
                                      true)->check(CLI::PositiveNumber);
        }
    },
    {
        CLIOptStr::INTERNAL_CORE,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::INTERNAL_CORE,
                                      app_settings->internal_thread_core,
                                      "CPU core affinity for Rivermax internal thread",
                                      true)->check(CLI::Range(CPU_NONE, MAX_CPU_RANGE));
        }
    },
    {
        CLIOptStr::APPLICATION_CORE,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::APPLICATION_CORE,
                                      app_settings->app_threads_cores,
                                      "CPU core affinities comma separated list for the application")
                                      ->delimiter(',')->check(CLI::Range(CPU_NONE, MAX_CPU_RANGE));
        }
    },
    {
        CLIOptStr::CHECKSUM_HEADER,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_flag(CLIOptStr::CHECKSUM_HEADER,
                                    app_settings->use_checksum_header,
                                    "Use checksum header");
        }
    },
    {
        CLIOptStr::WAIT_RETRY,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::WAIT_RETRY,
                                      app_settings->hw_queue_full_sleep_us,
                                      "Time to sleep in microseconds between subsequent "
                                      "commit retries if the HW queue is full")->check(CLI::PositiveNumber);
        }
    },
    {
        CLIOptStr::GPU_ID,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::GPU_ID,
                                      app_settings->gpu_id,
                                      "Enable GPU direct, value is GPU id")->check(CLI::Range(0, MAX_GPU_ID));
        }
    },
    {
        CLIOptStr::LOCK_GPU_CLOCKS,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::LOCK_GPU_CLOCKS,
                                    app_settings->lock_gpu_clocks,
                                    "Lock GPU clocks to their maximum frequency",
                                    true)->check(CLI::Range(0, 1));
        }
    },
    {
        CLIOptStr::ALLOCATOR_TYPE,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::ALLOCATOR_TYPE,
                                      app_settings->allocator_type,
                                      "Memory allocator type")
                                      ->transform(CLI::CheckedTransformer(UI_ALLOCATOR_TYPES));
        }
    },
    {
        CLIOptStr::REGISTER_MEMORY,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_flag(CLIOptStr::REGISTER_MEMORY,
                                    app_settings->register_memory,
                                    "Register memory on the application side for better performance");
        }
    },
    {
        CLIOptStr::HEADER_DATA_SPLIT,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_flag(CLIOptStr::HEADER_DATA_SPLIT,
                                    app_settings->header_data_split,
                                    "Use separate memory for headers and payloads");
        }
    },
    {
        CLIOptStr::APP_MEMORY_ALLOC,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_flag(CLIOptStr::APP_MEMORY_ALLOC,
                                    app_settings->app_memory_alloc,
                                    "I\\O Memory allocated by application");
        }
    },
    {
        CLIOptStr::ENABLE_STATS_READER,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_flag(CLIOptStr::ENABLE_STATS_READER,
                                    app_settings->enable_statistics_reader,
                                    "Enable statistics reader");
        }
    },
    {
        CLIOptStr::STATS_CORE,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::STATS_CORE,
                                      app_settings->statistics_reader_core,
                                      "CPU core affinity for statistics reader thread")
                                      ->check(CLI::Range(CPU_NONE, MAX_CPU_RANGE));
        }
    },
    {
        CLIOptStr::STATS_SESSION_ID,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::STATS_SESSION_ID,
                                      app_settings->session_id_stats,
                                      "Present runtime statistics of the given session id")
                                      ->check(CLI::NonNegativeNumber);
        }
    },
    {
        CLIOptStr::STATS_REPORT_INTERVAL,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::STATS_REPORT_INTERVAL,
                                      app_settings->stats_report_interval_ms,
                                      "Display statistics interval, ms",
                                      true)->check(CLI::NonNegativeNumber);
        }
    },
    {
        CLIOptStr::VIDEO_RESOLUTION,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::VIDEO_RESOLUTION,
                                      app_settings->media.resolution,
                                      "Video resolution in format <width>x<height>")
                                      ->check(VideoResolutionValidator())
                                      ->check(CLI::IsMember(SUPPORTED_VIDEO_RESOLUTIONS))
                                      ->default_val(Resolution(FHD_WIDTH, FHD_HEIGHT));
        }
    },
    {
        CLIOptStr::VIDEO_FRAME_RATE,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::VIDEO_FRAME_RATE,
                                      app_settings->media.frame_rate,
                                      "Video frame rate in format <numerator>/<denominator> or <integer>")
                                      ->check(VideoFrameRateValidator())
                                      ->check(CLI::IsMember(SUPPORTED_VIDEO_FRAME_RATES))
                                      ->default_val(60);
        }
    },
    {
        CLIOptStr::VIDEO_SAMPLING,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::VIDEO_SAMPLING,
                                      app_settings->media.sampling_type,
                                      "Video sampling type")
                                      ->transform(CLI::CheckedTransformer(create_mapping_vector(
                                                  SUPPORTED_VIDEO_SAMPLING_TYPES), CLI::ignore_case))
                                      ->default_val(enum_to_string(VideoSampling::YCbCr_4_2_2));
        }
    },
    {
        CLIOptStr::VIDEO_BIT_DEPTH,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::VIDEO_BIT_DEPTH,
                                      app_settings->media.bit_depth,
                                      "Video bit depth")
                                      ->transform(CLI::CheckedTransformer(create_mapping_vector(
                                                  SUPPORTED_VIDEO_BIT_DEPTHS), CLI::ignore_case))
                                      ->default_val(enum_to_string(ColorBitDepth::_10));
        }
    },
    {
        CLIOptStr::VIDEO_FILE,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::VIDEO_FILE,
                                      app_settings->video_file,
                                      "Video file to send")
                                      ->check(CLI::ExistingFile);
        }
    },
    {
        CLIOptStr::DYNAMIC_FILE_LOADING,
        [](CLI::App_p parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_flag(CLIOptStr::DYNAMIC_FILE_LOADING,
                                      app_settings->dynamic_video_file_load,
                                      "Enable dynamic file loading: Load video frames on the fly during transmission)");
        }
    },
};
