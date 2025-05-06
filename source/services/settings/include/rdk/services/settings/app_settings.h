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

#ifndef RDK_SERVICES_SETTINGS_APP_SETTINGS_H_
#define RDK_SERVICES_SETTINGS_APP_SETTINGS_H_

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

#include "rdk/services/utils/defs.h"
#include "rdk/services/media/media_defs.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Application settings.
 *
 * The struct will be used to hold application parameters required
 * for the application to operate as requested by the user.
 */
struct AppSettings
{
    int sleep_between_operations_us;
    bool sleep_between_operations;
    std::string local_ip;
    std::string local_mac;
    std::vector<std::string> local_ips;
    std::string source_ip;
    std::vector<std::string> source_ips;
    uint16_t source_port;
    std::string destination_ip;
    std::vector<std::string> destination_ips;
    uint16_t destination_port;
    std::vector<uint16_t> destination_ports;
    uint16_t num_of_total_streams;
    size_t num_of_total_flows;
    size_t num_of_threads;
    int internal_thread_core;
    std::vector<int> app_threads_cores;
    bool print_parameters;
    PacketPacingRate rate;
    size_t num_of_memory_blocks;
    size_t num_of_chunks_in_mem_block;
    size_t num_of_chunks;
    size_t num_of_packets_in_chunk;
    bool num_of_packets_in_chunk_specified;
    size_t num_of_packets_in_mem_block;
    uint16_t packet_payload_size;
    uint16_t packet_app_header_size;
    bool use_checksum_header;
    uint32_t hw_queue_full_sleep_us;
    int gpu_id;
    bool lock_gpu_clocks;
    AllocatorTypeUI allocator_type;
    MediaSettings media;
    bool ref_clk_is_ptp;
    uint32_t session_id_stats;
    bool register_memory;
    bool header_data_split;
    bool app_memory_alloc;
    bool enable_statistics_reader;
    int statistics_reader_core;
    uint32_t stats_report_interval_ms;
    std::string video_file;
    bool dynamic_video_file_load;
public:
    /**
     * @brief: Default destructor.
     */
    virtual ~AppSettings() = default;
    /**
     * @brief: Virtual initialization method.
     *
     * The method will be used to initialize the default values of the
     * application settings. The method can be overridden by the
     * derived classes to set the default values of the specific
     * application settings.
     */
    virtual void init_default_values();
    /**
     * @brief: Factory method for a fully initialized instance.
     *
     * @tparam DerivedSettingsType: The derived type that inherits from AppSettings.
     *
     * @return: Shared pointer to the fully initialized instance.
     */
    template<typename DerivedSettingsType>
    static std::shared_ptr<DerivedSettingsType> create_default() {
        static_assert(std::is_base_of<AppSettings, DerivedSettingsType>::value,
            "DerivedSettingsType must inherit from AppSettings");
        auto settings = std::make_shared<DerivedSettingsType>();
        settings->init_default_values();
        return settings;
    }
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_SETTINGS_APP_SETTINGS_H_ */
