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

#include <memory>
#include <cstring>
#include <map>

#include "CLI/CLI.hpp"
#include "rt_threads.h"

#include "services/utils/defs.h"
#include "services/cli/options.h"
#include "services/cli/cli_manager.h"
#include "services/cli/validators.h"

using namespace ral::lib::services;

static const std::map<std::string, AllocatorTypeUI> UI_ALLOCATOR_TYPES{
    { "auto",           AllocatorTypeUI::Auto },
    { "malloc",         AllocatorTypeUI::Malloc },
    { "hugepage",       AllocatorTypeUI::HugePageDefault },
    { "hugepage-2m",    AllocatorTypeUI::HugePage2MB },
    { "hugepage-512m",  AllocatorTypeUI::HugePage512MB },
    { "hugepage-1g",    AllocatorTypeUI::HugePage1GB },
    { "gpu",            AllocatorTypeUI::Gpu },
};

/**
 * @brief: CLI options factory map.
 *
 * The map creates CLI options based on a given key.
 * It maps from the key to a dedicated lambda function that creates the CLI option and adds it
 * to the parser object.
 *
 * Future CLI options, that are common enough, can be added to this map and used using
 * @ref ral::lib::services::CLIParserManager::add_option method.
 */
cli_opt_factory_map_t CLIParserManager::s_cli_opt_fuctory {
    {
        CLIOptStr::LOCAL_IP,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::LOCAL_IP,
                                      app_settings->source_ip,
                                      "Local IP of the NIC")->check(CLI::ValidIPV4)->required();
        }
    },
    {
        CLIOptStr::LOCAL_IPS,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
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
        CLIOptStr::SRC_IPS,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
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
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::DST_IP,
                                      app_settings->destination_ip,
                                      "Destination IP of the connection",
                                      true)->check(CLI::ValidIPV4);
        }
    },
    {
        CLIOptStr::DST_IPS,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
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
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::DST_PORT,
                                      app_settings->destination_port,
                                      "Destination port of the connection",
                                      true)->check(CLI::Range(MIN_PORT, MAX_PORT));
        }
    },
    {
        CLIOptStr::DST_PORTS,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
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
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::THREADS,
                                      app_settings->num_of_threads,
                                      "Number of threads to use",
                                      true)->check(CLI::PositiveNumber);
        }
    },
    {
        CLIOptStr::FLOWS,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
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
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
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
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_flag(CLIOptStr::VERBOSE,
                                    app_settings->print_parameters,
                                    "Print verbose info");
        }
    },
    {
        CLIOptStr::CHUNKS,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::CHUNKS,
                                      app_settings->num_of_chunks,
                                      "Number of memory chunks",
                                      true)->check(CLI::PositiveNumber);
        }
    },
    {
        CLIOptStr::PACKETS,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::PACKETS,
                                      app_settings->num_of_packets_in_chunk,
                                      "Number of packets in chunk",
                                      true)->check(CLI::PositiveNumber);
        }
    },
    {
        CLIOptStr::PAYLOAD_SIZE,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::PAYLOAD_SIZE,
                                      app_settings->packet_payload_size,
                                      "Packet's payload size",
                                      true)->check(CLI::PositiveNumber);
        }
    },
    {
        CLIOptStr::APP_HDR_SIZE,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::APP_HDR_SIZE,
                                      app_settings->packet_app_header_size,
                                      "Packet's application header size",
                                      true)->check(CLI::PositiveNumber);
        }
    },
    {
        CLIOptStr::SLEEP_US,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::SLEEP_US,
                                      app_settings->sleep_between_operations_us,
                                      "Sleep duration in microsecond after processing all thread's streams",
                                      true)->check(CLI::PositiveNumber);
        }
    },
    {
        CLIOptStr::SLEEP,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_flag(CLIOptStr::SLEEP,
                                    app_settings->sleep_between_operations,
                                    "Don't block on send, do sleep");
        }
    },
    {
        CLIOptStr::RATE_BPS,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::RATE_BPS,
                                      app_settings->rate.bps,
                                      "Rate limit in bits per second per flow")->check(CLI::PositiveNumber);
        }
    },
    {
        CLIOptStr::RATE_BURST,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::RATE_BURST,
                                      app_settings->rate.max_burst_in_packets,
                                      "Maximum allowed burst size in number of packets per flow",
                                      true)->check(CLI::PositiveNumber);
        }
    },
    {
        CLIOptStr::INTERNAL_CORE,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::INTERNAL_CORE,
                                      app_settings->internal_thread_core,
                                      "CPU core affinity for Rivermax internal thread",
                                      true)->check(CLI::Range(CPU_NONE, MAX_CPU_RANGE));
        }
    },
    {
        CLIOptStr::APPLICATION_CORE,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::APPLICATION_CORE,
                                      app_settings->app_threads_cores,
                                      "CPU core affinities comma separated list for the application")
                                      ->delimiter(',')->check(CLI::Range(CPU_NONE, MAX_CPU_RANGE));
        }
    },
    {
        CLIOptStr::CHECKSUM_HEADER,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_flag(CLIOptStr::CHECKSUM_HEADER,
                                    app_settings->use_checksum_header,
                                    "Use checksum header");
        }
    },
    {
        CLIOptStr::WAIT_RETRY,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::WAIT_RETRY,
                                      app_settings->hw_queue_full_sleep_us,
                                      "Time to sleep in microseconds between subsequent "
                                      "commit retries if the HW queue is full")->check(CLI::PositiveNumber);
        }
    },
    {
        CLIOptStr::GPU_ID,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::GPU_ID,
                                      app_settings->gpu_id,
                                      "Enable GPU direct, value is GPU id")->check(CLI::Range(0, MAX_GPU_ID));
        }
    },
    {
        CLIOptStr::ALLOCATOR_TYPE,
        [](std::shared_ptr<CLI::App> parser, std::shared_ptr<AppSettings> app_settings)
        {
            return parser->add_option(CLIOptStr::ALLOCATOR_TYPE,
                                      app_settings->allocator_type,
                                      "Memory allocator type")
                                      ->transform(CLI::Transformer(UI_ALLOCATOR_TYPES));
        }
    },
};
