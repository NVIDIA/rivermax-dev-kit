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

#include "rt_threads.h"

#include "rdk/services/utils/defs.h"
#include "rdk/services/cli/options.h"
#include "rdk/services/cpu/affinity/rivermax_affinity.h"

using namespace rivermax::dev_kit::services;

constexpr int INVALID_CORE_NUMBER = -1;

void AppSettings::init_default_values()
{
    destination_ip = DESTINATION_IP_DEFAULT;
    destination_port = DESTINATION_PORT_DEFAULT;
    num_of_threads = NUM_OF_THREADS_DEFAULT;
    num_of_total_streams = NUM_OF_TOTAL_STREAMS_DEFAULT;
    num_of_total_flows = NUM_OF_TOTAL_FLOWS_DEFAULT;
    internal_thread_core = CPU_NONE;
    app_threads_cores = std::vector<int>(num_of_threads, CPU_NONE);
    rate = { 0, 0 };
    num_of_chunks = NUM_OF_CHUNKS_DEFAULT;
    num_of_packets_in_chunk = NUM_OF_PACKETS_IN_CHUNK_DEFAULT;
    packet_payload_size = PACKET_PAYLOAD_SIZE_DEFAULT;
    packet_app_header_size = PACKET_APP_HEADER_SIZE_DEFAULT;
    sleep_between_operations_us = SLEEP_BETWEEN_OPERATIONS_US_DEFAULT;
    sleep_between_operations = false;
    print_parameters = false;
    use_checksum_header = false;
    hw_queue_full_sleep_us = 0;
    gpu_id = INVALID_GPU_ID;
    lock_gpu_clocks = LOCK_GPU_CLOCKS_DEFAULT;
    allocator_type = AllocatorTypeUI::Auto;
    session_id_stats = UINT_MAX;
    ref_clk_is_ptp = true;
    register_memory = false;
    header_data_split = false;
    app_memory_alloc = false;
    enable_statistics_reader = false;
    statistics_reader_core = INVALID_CORE_NUMBER;
    stats_report_interval_ms = STATS_REPORT_INTERVAL_MS_DEFAULT;
    dynamic_video_file_load = false;
}
