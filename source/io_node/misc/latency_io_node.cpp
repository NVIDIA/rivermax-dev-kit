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

#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>
#include <cstring>

#include <rivermax_api.h>
#include "rt_threads.h"

#include "rdk/io_node/misc/latency_io_node.h"
#include "rdk/services/cpu/affinity/affinity.h"

using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

void LatencyStats::update(int64_t value) {
    m_sumval += value;
    if (value > m_maxval) {
        m_maxval = value;
    }
    if (value < m_minval) {
        m_minval = value;
    }
    if (m_percentiles.size()) {
        if (m_count < MAX_SAMPLES) {
            m_samples.push_back(value);
        } else {
            m_too_many_samples = true;
        }
    }
    m_count++;
}

void LatencyStats::reset() {
    m_minval = DEFAULT_MIN_VALUE;
    m_maxval = -DEFAULT_MIN_VALUE;
    m_sumval = 0;
    m_count = 0;
    m_too_many_samples = false;
    if (m_percentiles.size()) {
        m_samples.clear();
        m_samples.reserve(MAX_SAMPLES);
    }
}

void LatencyStats::calc_percentiles()
{
    std::sort(m_samples.begin(), m_samples.end());
    m_perc_values.clear();
    m_perc_values.reserve(m_percentiles.size());
    if (m_percentiles.size()) {
        if (m_samples.size() == 0) {
            return;
        }
        for (auto p : m_percentiles) {
            if (p < 0.0 || p > 1.0) {
                break;
            }
            size_t pos;
            if (m_samples.size() == 1) {
                pos = 0;
            } else {
                pos = static_cast<size_t>(p * (m_samples.size() - 1) + 0.5);
            }
            m_perc_values.push_back(m_samples[pos]);
        }
    }
}

std::ostream& LatencyStats::print(std::ostream& out) const
{
    if (m_count == 0) {
        out << m_name << ": no values" << std::endl;
        return out;
    }

    std::ios old_state(nullptr);
    old_state.copyfmt(out);

    out << m_name << ":\n"
        << "  Avg: " << get_avg() << "\n"
        << "  Min: " << get_min() << "\n"
        << "  Max: " << get_max() << "\n";

    if (m_perc_values.size()) {
        out << "Total " << m_count << " samples\n";
        if (m_too_many_samples) {
            out << "Percentiles calculated on first " << MAX_SAMPLES << " samples\n";
        }
        out.precision(5);
        auto p = m_percentiles.begin();
        for (auto p_val : m_perc_values) {
            out << " " << std::setw(9) << (*p++) * 100 << "% : " << p_val << "\n";
        }
        out << "-----\n";
    }

    out.copyfmt(old_state);
    return out;
}

LatencyIONode::LatencyIONode(
    const LatencyNodeSettings& settings,
    std::shared_ptr<MemoryUtils> header_mem_utils,
    std::shared_ptr<MemoryUtils> payload_mem_utils,
    time_handler_ns_cb_t time_handler_cb
    ) :
    m_send_flow(0, settings.app->destination_ip, settings.app->destination_port),
    m_receive_flow(ReceiveFlow(0, "0.0.0.0", 0, settings.receive_ip, settings.receive_port)),
    m_network_address(ReceiveFlow(0, settings.app->local_ip, settings.app->source_port,
        settings.app->destination_ip, settings.app->destination_port)),
    m_sleep_between_operations_us(settings.app->sleep_between_operations_us),
    m_print_parameters(settings.app->print_parameters),
    m_send_header_region{nullptr, 0, 0},
    m_send_payload_region{nullptr, 0, 0},
    m_receive_header_region{nullptr, 0, 0},
    m_receive_payload_region{nullptr, 0, 0},
    m_gpu_direct_tx(settings.gpu_direct_tx),
    m_gpu_direct_rx(settings.gpu_direct_rx),
    m_cpu_core_affinity(settings.app->app_threads_cores[0]),
    m_client_mode(settings.client_mode),
    m_measure_interval_sec(settings.measure_interval),
    m_header_mem_utils(std::move(header_mem_utils)),
    m_payload_mem_utils(std::move(payload_mem_utils)),
    m_get_time_ns_cb(std::move(time_handler_cb)),
    m_percentiles(settings.percentiles)
{
}

void LatencyIONode::set_cpu_resources()
{
    set_current_thread_affinity(m_cpu_core_affinity);
    rt_set_thread_priority(RMAX_THREAD_PRIORITY_TIME_CRITICAL - 1);
}

void LatencyIONode::operator()()
{
    set_cpu_resources();
    ReturnStatus rc = create_send_stream();

    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to create sender stream !!!!!" << std::endl;
        return;
    }
    print_parameters();
    prepare_send_buffer();

    rc = create_receive_stream();
    if (rc != ReturnStatus::success) {
        return;
    }
    rc = attach_receive_flow();
    if (rc != ReturnStatus::success) {
        return;
    }

    std::cout << "Main measurement loop started" << std::endl;
    if (m_client_mode) {
        send_receive();
    } else {
        receive_send();
    }

    detach_receive_flow();
    destroy_receive_stream();
    destroy_send_stream();
}
