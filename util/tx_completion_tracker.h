/*
 * SPDX-FileCopyrightText: Copyright (c) 2001-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef _TX_COMPLETION_TRACKER_H_
#define _TX_COMPLETION_TRACKER_H_

#include "rivermax_api.h"
#include "rivermax_defs.h"
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

template <typename T, 
    rmx_status (*mark_for_tracking)(const T*, uint64_t),
    rmx_status (*poll_for_completion)(const T*),
    const rmx_output_chunk_completion *(*get_last_completion)(const T*)>
class TxCompletionTracker {
public:
    class Exception : public std::exception
    {
    private:
        std::string m_error;
    public:
        Exception(const std::string& error): m_error {error} {}
        const char* what() const noexcept override { return m_error.c_str(); }
    };

private:
    static constexpr size_t every_nth_completion_2_print {100000};
    T &m_handle;
    uint64_t m_next_token_to_send;
    uint64_t m_next_token_to_complete;
    uint64_t m_last_timestamp;

public:
    TxCompletionTracker() = delete;
    TxCompletionTracker(T &chunk_handle)
        : m_handle {chunk_handle} 
        , m_next_token_to_send {0}
        , m_next_token_to_complete {0}
        , m_last_timestamp {0}
    {}
    size_t CountIncomplete() const { return (m_next_token_to_send - m_next_token_to_complete); }
    void TrackCommit()
    {
        auto status = mark_for_tracking(&m_handle, ++m_next_token_to_send);
        if (status != RMX_OK) {
            throw Exception("Failed to mark chunk for tracking: " + std::to_string(status));
        }
    }
    void TrackCompletion() {
        rmx_status status;
        do {
            status = poll_for_completion(&m_handle);
        } while (status != RMX_OK);

        auto completion = get_last_completion(&m_handle);
        if (++m_next_token_to_complete != rmx_output_get_completion_user_token(completion)) {
            throw Exception("Out of order completion. " 
                "Expected: " + std::to_string(m_next_token_to_complete) + ". " +
                "Received: " + std::to_string(rmx_output_get_completion_user_token(completion)));
        }
        if (m_last_timestamp > rmx_output_get_completion_timestamp(completion)) {
            auto message = "Out of order completion timestamp: " 
                "Prev=" + std::to_string(m_last_timestamp) + ", " +
                "New=" + std::to_string(rmx_output_get_completion_timestamp(completion)) + ", " +
                "delta=" + std::to_string(m_last_timestamp - rmx_output_get_completion_timestamp(completion)) + ".";
            throw Exception(message);
        }
        m_last_timestamp = rmx_output_get_completion_timestamp(completion);
    }

    void print(std::ostream &output) const {
        output << "completion: #" << m_next_token_to_complete << " TS: " << m_last_timestamp << std::endl;
    }
};

#endif //_TX_COMPLETION_TRACKER_H_
