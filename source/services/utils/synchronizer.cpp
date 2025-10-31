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

#include "rdk/services/utils/synchronizer.h"

using namespace rivermax::dev_kit::services;

void LinearSynchronizer::reset() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_request_count = 0;
    m_new_candidate_time = 0;
    m_sync_status = ReturnStatus::success;
    m_checkers.clear();
}

ReturnStatus LinearSynchronizer::request(uint64_t requested_time, std::function<int(uint64_t)> checker,
    uint64_t &consensus_time)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_new_candidate_time = std::max(m_new_candidate_time, requested_time);
    m_checkers.push_back(checker);
    m_request_count++;

    if (m_request_count < m_num_of_threads) {
        m_cv.wait(lock, [this] { return m_request_count == m_num_of_threads; });
    } else {
        uint64_t proposed_time = m_new_candidate_time;
        bool all_zero = true;
        bool any_negative = false;

        do {
            all_zero = true;
            any_negative = false;
            int max_checker_result = 0;

            for (const auto& checker_function : m_checkers) {
                int checker_result = checker_function(proposed_time);
                if (checker_result == 0) {
                    continue;
                } else if (checker_result < 0) {
                    any_negative = true;
                    break;
                } else {
                    all_zero = false;
                    max_checker_result = std::max(max_checker_result, checker_result);
                }
            }
            if (any_negative) {
                m_sync_status = ReturnStatus::failure;
            } else if (!all_zero) {
                proposed_time += max_checker_result;
            }
        } while (!all_zero && !any_negative);
        m_new_candidate_time = proposed_time;
        m_cv.notify_all();
    }

    consensus_time = m_new_candidate_time;
    return m_sync_status;
}
