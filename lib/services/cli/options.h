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

#ifndef RMAX_APPS_LIB_LIB_SERVICES_CLI_OPTIONS_H_

#include <climits>
#include <unordered_map>
#include <functional>
#include <memory>

#include "CLI/CLI.hpp"

#include "services/utils/defs.h"

namespace ral
{
namespace lib
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

typedef std::unordered_map<
    std::string, std::function<CLI::Option* (std::shared_ptr<CLI::App>, std::shared_ptr<AppSettings>)>>
    cli_opt_factory_map_t;

/**
 * @brief: CLI options string key.
 *
 * This class should be used as a key to @ref ral::lib::services::CLIParserManager::s_cli_opt_fuctory.
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
    static const char* ALLOCATOR_TYPE;
};


} // namespace services
} // namespace lib
} // namespace ral


#define RMAX_APPS_LIB_LIB_SERVICES_CLI_OPTIONS_H_
#endif /* RMAX_APPS_LIB_LIB_SERVICES_CLI_OPTIONS_H_ */
