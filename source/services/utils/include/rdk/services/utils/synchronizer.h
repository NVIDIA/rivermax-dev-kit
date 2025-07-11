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

#ifndef RDK_SERVICES_UTILS_SYNCHRONIZER_H_
#define RDK_SERVICES_UTILS_SYNCHRONIZER_H_

#include <cstddef>

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "rdk/services/error_handling/return_status.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Interface for thread synchronization mechanisms.
 *
 * This abstract class defines the interface that all synchronizer implementations
 * must implement. It provides a method for requesting synchronization between
 * multiple threads with a consensus time mechanism.
 */
class ISynchronizer
{
public:
    /**
     * @brief: Virtual destructor.
     */
    virtual ~ISynchronizer() = default;
    /**
     * @brief: Requests synchronization with other threads.
     *
     * This pure virtual method must be implemented by derived classes to provide
     * thread synchronization functionality. It allows threads to request a consensus
     * time and uses a checker function to validate the proposed time.
     *
     * @param [in] requested_time: The time requested by the calling thread.
     * @param [in] checker: Function that validates the proposed time and returns adjustment values.
     * @param [out] consensus_time: The agreed-upon consensus time for all threads.
     *
     * @return: Status of the synchronization operation.
     */
    virtual ReturnStatus request(uint64_t requested_time, std::function<int(uint64_t)> checker, uint64_t& consensus_time) = 0;
};

/**
 * @brief: Thread synchronization implementation using condition variables.
 *
 * This class implements a thread synchronization mechanism that allows multiple
 * threads to reach consensus on a common time value. It uses mutexes and condition
 * variables to coordinate between threads and ensures all threads wait for each
 * other before proceeding.
 *
 * The synchronizer works by:
 * 1. Collecting time requests from all threads
 * 2. Finding the maximum requested time as a starting point
 * 3. Using a checker function to validate and adjust the proposed time
 * 4. Reaching consensus when all threads agree on the final time
 */
class Synchronizer : public ISynchronizer
{
public:
    /**
     * @brief: Synchronizer constructor.
     *
     * @param [in] numThreads: Number of threads that will participate in synchronization.
     */
    Synchronizer(int numThreads) :
        m_num_of_threads(numThreads),
        m_request_count(0),
        m_new_candidate(0),
        m_sync_status(ReturnStatus::success) {}
    /**
     * @brief: Virtual destructor.
     */
     virtual ~Synchronizer() = default;
    /**
     * @brief: Resets the synchronizer state.
     *
     * This method resets the internal counters and status to allow the synchronizer
     * to be reused for another round of synchronization.
     */
    void reset() {
        m_request_count = 0;
        m_new_candidate = 0;
        m_sync_status = ReturnStatus::success;
    }
    /**
     * @brief: Requests synchronization with other threads.
     *
     * This method implements the thread synchronization logic. It collects time
     * requests from all participating threads and uses a checker function to
     * validate and adjust the proposed consensus time.
     *
     * The synchronization process:
     * 1. Each thread calls this method with their requested time
     * 2. The method waits until all threads have made their requests
     * 3. The checker function is called for each thread to validate/adjust the time
     * 4. A consensus time is reached when all threads agree
     *
     * @param [in] requested_time: The time requested by the calling thread.
     * @param [in] checker: Function that validates the proposed time and returns adjustment values.
     *                      Returns 0 if the time is acceptable, positive for required increments,
     *                      or negative if the time is invalid.
     * @param [out] consensus_time: The agreed-upon consensus time for all threads.
     *
     * @return: Status of the synchronization operation.
     */
    ReturnStatus request(uint64_t requested_time,
                        std::function<int(uint64_t)> checker,
                        uint64_t &consensus_time) override 
    {
        std::unique_lock<std::mutex> lock(m_mtx);
        m_new_candidate = std::max(m_new_candidate, requested_time);
        m_request_count++;

        if (m_request_count < m_num_of_threads) {
            m_cv.wait(lock, [this] { return m_request_count == m_num_of_threads; });
        } else {
            uint64_t proposed_time = m_new_candidate;
            bool all_zero = true;
            bool any_negative = false;

            do {
                all_zero = true;
                any_negative = false;
                int max_checker_result = 0;

                for (int i = 0; i < m_num_of_threads; ++i) {
                    int checker_result = checker(proposed_time);
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
            m_new_candidate = proposed_time;
            m_cv.notify_all();
        }

        consensus_time = m_new_candidate;
        return m_sync_status;
    }

private:
    /* Number of threads participating in synchronization */
    int m_num_of_threads;
    /* Current count of threads that have made synchronization requests */
    int m_request_count;
    /* The current candidate time for consensus */
    uint64_t m_new_candidate;
    /* Mutex for protecting shared state */
    std::mutex m_mtx;
    /* Condition variable for thread coordination */
    std::condition_variable m_cv;
    /* Status of the synchronization operation */
    ReturnStatus m_sync_status;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif // RDK_SERVICES_UTILS_SYNCHRONIZER_H_
