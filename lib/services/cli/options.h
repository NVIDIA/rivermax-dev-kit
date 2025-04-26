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
    static constexpr const char* LOCAL_IP = "-l,--local-ip";
    static constexpr const char* LOCAL_IPS = "-l,--local-ips";
    static constexpr const char* SRC_IPS = "-s,--src-ips";
    static constexpr const char* DST_IP = "-d,--dst-ip";
    static constexpr const char* DST_IPS = "-d,--dst-ips";
    static constexpr const char* DST_PORT = "-p,--dst-port";
    static constexpr const char* DST_PORTS = "-p,--dst-ports";
    static constexpr const char* THREADS = "-T,--threads";
    static constexpr const char* FLOWS = "-F,--flows";
    static constexpr const char* STREAMS = "-S,--streams";
    static constexpr const char* VERBOSE = "-v,--verbose";
    static constexpr const char* CHUNKS = "-C,--chunks";
    static constexpr const char* PACKETS = "-K,--packets";
    static constexpr const char* PAYLOAD_SIZE = "-y,--payload-size";
    static constexpr const char* APP_HDR_SIZE = "-e,--app-hdr-size";
    static constexpr const char* SLEEP_US = "-z,--sleep-us";
    static constexpr const char* SLEEP = "-z,--sleep";
    static constexpr const char* RATE_BPS = "-r,--rate-bps";
    static constexpr const char* RATE_BURST = "-b,--rate-burst";
    static constexpr const char* INTERNAL_CORE = "-i,--internal-core";
    static constexpr const char* APPLICATION_CORE = "-a,--application-core";
    static constexpr const char* CHECKSUM_HEADER = "-x,--checksum-header";
    static constexpr const char* WAIT_RETRY = "-w,--wait-retry";
    static constexpr const char* GPU_ID = "-g,--gpu-id";
};


} // namespace services
} // namespace lib
} // namespace ral


#define RMAX_APPS_LIB_LIB_SERVICES_CLI_OPTIONS_H_
#endif /* RMAX_APPS_LIB_LIB_SERVICES_CLI_OPTIONS_H_ */
