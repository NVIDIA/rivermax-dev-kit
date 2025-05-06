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

#ifndef RDK_SERVICES_CLI_OPTIONS_H_
#define RDK_SERVICES_CLI_OPTIONS_H_

#include <climits>
#include <unordered_map>
#include <functional>
#include <memory>

#include "CLI/CLI.hpp"

#include "rdk/services/settings/app_settings.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

constexpr uint16_t MIN_PORT = 1;
constexpr uint16_t MAX_PORT = 65535;
constexpr size_t MIN_NUM_OF_TOTAL_FLOWS = 1;
constexpr int IP_OCTET_LEN = 256;
constexpr size_t MAX_NUM_OF_TOTAL_FLOWS = MAX_PORT * IP_OCTET_LEN;
constexpr int INVALID_GPU_ID = INT_MAX;
constexpr int MAX_GPU_ID = 500;
constexpr bool LOCK_GPU_CLOCKS_DEFAULT = true;
constexpr size_t NUM_OF_THREADS_DEFAULT = 1;
constexpr uint16_t NUM_OF_TOTAL_STREAMS_DEFAULT = 1;
constexpr size_t NUM_OF_TOTAL_FLOWS_DEFAULT = 1;
constexpr size_t NUM_OF_CHUNKS_DEFAULT = 1;
constexpr size_t NUM_OF_PACKETS_IN_CHUNK_DEFAULT = 4096;
constexpr uint16_t PACKET_PAYLOAD_SIZE_DEFAULT = 1460;
constexpr uint16_t PACKET_APP_HEADER_SIZE_DEFAULT = 0;
constexpr int SLEEP_BETWEEN_OPERATIONS_US_DEFAULT = 0;
constexpr const char* DESTINATION_IP_DEFAULT = "234.5.6.7";
constexpr uint16_t DESTINATION_PORT_DEFAULT = 50000;
constexpr uint32_t STATS_REPORT_INTERVAL_MS_DEFAULT = 1000;

typedef std::unordered_map<
    std::string, std::function<CLI::Option* (CLI::App_p, std::shared_ptr<AppSettings>)>>
    cli_opt_factory_map_t;

/**
 * @brief: CLI options string key.
 *
 * This class should be used as a key to @ref CLIParserManager::s_cli_opt_fuctory.
 */
class CLIOptStr
{
public:
    static const char* LOCAL_IP;
    static const char* LOCAL_IPS;
    static const char* SRC_IP;
    static const char* SRC_IPS;
    static const char* DST_IP;
    static const char* DST_IPS;
    static const char* DST_PORT;
    static const char* DST_PORTS;
    static const char* THREADS;
    static const char* FLOWS;
    static const char* STREAMS;
    static const char* VERBOSE;
    static const char* CHUNKS;
    static const char* PACKETS;
    static const char* PAYLOAD_SIZE;
    static const char* APP_HDR_SIZE;
    static const char* SLEEP_US;
    static const char* SLEEP;
    static const char* RATE_BPS;
    static const char* RATE_BURST;
    static const char* INTERNAL_CORE;
    static const char* APPLICATION_CORE;
    static const char* CHECKSUM_HEADER;
    static const char* WAIT_RETRY;
    static const char* GPU_ID;
    static const char* LOCK_GPU_CLOCKS;
    static const char* ALLOCATOR_TYPE;
    static const char* REGISTER_MEMORY;
    static const char* HEADER_DATA_SPLIT;
    static const char* APP_MEMORY_ALLOC;
    static const char* ENABLE_STATS_READER;
    static const char* STATS_CORE;
    static const char* STATS_SESSION_ID;
    static const char* STATS_REPORT_INTERVAL;
    static const char* VIDEO_RESOLUTION;
    static const char* VIDEO_FRAME_RATE;
    static const char* VIDEO_SAMPLING;
    static const char* VIDEO_BIT_DEPTH;
    static const char* VIDEO_FILE;
    static const char* DYNAMIC_FILE_LOADING;
};
/**
 * @brief: CLI groups string key.
 *
 * This class should be used as a key for CLI groups.
 */
class CLIGroupStr
{
public:
    static const char* VIDEO_FORMAT_OPTIONS;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_CLI_OPTIONS_H_ */
