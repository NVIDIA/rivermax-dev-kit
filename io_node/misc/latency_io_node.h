/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef RMAX_APPS_LIB_IO_NODE_MISC_LATENCY_IO_NODE_H_
#define RMAX_APPS_LIB_IO_NODE_MISC_LATENCY_IO_NODE_H_

#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>

#include <rivermax_api.h>

#include "api/rmax_apps_lib_api.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

namespace ral
{
namespace io_node
{
/**
 * @brief: Application constants.
 */
constexpr size_t DEFAULT_NUM_OF_SEND_CHUNKS = 4;
constexpr size_t DEFAULT_NUM_OF_RECEIVE_CHUNKS = 4;
constexpr size_t DEFAULT_SEND_BATCH_SIZE = 1;
constexpr size_t DEFAULT_SKIP_PACKETS = 4;
constexpr size_t DEFAULT_RESPONSE_SIZE = 1200;
constexpr size_t MAX_RX_CHUNK_SIZE = 1000;
constexpr size_t RTP_HEADER_SIZE = 20;

/**
* @brief: Latency statistics counter.
*/
class LatencyStats {
public:
    LatencyStats(const std::string name,
                 const std::vector<double>& percentiles = std::vector<double>()) :
        m_name(name), m_percentiles(percentiles), m_perc_values() { reset(); }
    virtual ~LatencyStats() = default;
    void update(int64_t value);
    void reset();
    int64_t get_cnt() const {return m_count;}
    int64_t get_min() const {return m_minval;}
    int64_t get_max() const {return m_maxval;}
    int64_t get_avg() const {return m_sumval / m_count;}
    void calc_percentiles();
    /**
     * @brief: Prints statistics values to an output stream.
     *
     * The method prints the statistical values to an output stream.
     *
     * @param [out] out: Output stream parameter print to.
     *
     * @return: Reference to the output stream.
     */
    virtual std::ostream& print(std::ostream& out) const;
    /**
     * @brief: Overrides operator << for @ref ral::io_node::LatencyStats reference.
     */
    friend std::ostream& operator<<(std::ostream& out, const LatencyStats& counter)
    {
        counter.print(out);
        return out;
    }
    /**
     * @brief: Overrides operator << for @ref ral::io_node::LatencyStats pointer.
     */
    friend std::ostream& operator<<(std::ostream& out, const LatencyStats* counter)
    {
        counter->print(out);
        return out;
    }

protected:
    static constexpr int64_t DEFAULT_MIN_VALUE = 999999999999999999;
    static constexpr int64_t MAX_SAMPLES = 10000000;
    const std::string m_name;
    int64_t m_minval;
    int64_t m_maxval;
    int64_t m_sumval;
    int64_t m_count;
    std::vector<int64_t> m_samples;
    std::vector<double> m_percentiles;
    std::vector<int64_t> m_perc_values;
    bool m_too_many_samples;
};

/**
 * @brief: Latency node settings.
 */
struct LatencyNodeSettings {
    std::shared_ptr<AppSettings> app;
    std::string receive_ip;
    uint16_t receive_port;
    bool client_mode;
    bool gpu_direct_tx;
    bool gpu_direct_rx;
    size_t measure_interval;
    bool track_completions;
    std::vector<double> percentiles;
};

/**
 * @brief: LatencyIONode class.
 *
 * This class implements a skeleton for a latency measurement process
 * that will be run under a std::thread by overriding the operator ().
 * The class defines the common parameters and resource
 * initialization/deintialization sequences for various latency
 * measurement modes implemented by descendent calsses.
 */
class LatencyIONode
{
protected:
    TwoTupleFlow m_send_flow;
    FourTupleFlow m_receive_flow;
    FourTupleFlow m_network_address;
    int m_sleep_between_operations_us;
    bool m_print_parameters;
    rmx_mem_region m_send_header_region;
    rmx_mem_region m_send_payload_region;
    rmx_mem_region m_receive_header_region;
    rmx_mem_region m_receive_payload_region;
    bool m_gpu_direct_tx;
    bool m_gpu_direct_rx;
    rmax_cpu_set_t m_cpu_affinity_mask;
    int m_cpu_core_affinity;
    bool m_client_mode;
    size_t m_measure_interval_sec;
    std::shared_ptr<MemoryUtils> m_header_mem_utils;
    std::shared_ptr<MemoryUtils> m_payload_mem_utils;
    time_handler_ns_cb_t m_get_time_ns_cb;
    std::vector<double> m_percentiles;
public:
    /**
     * @brief: LatencyIONode constructor.
     *
     * @param [in] settings: Latency measurement settings.
     * @param [in] header_mem_utils: Header Memory utilities.
     * @param [in] payload_mem_utils: Payload Memory utilities.
     * @param [in] time_handler_cb: Timer handler.
     */
    LatencyIONode(
        const LatencyNodeSettings& settings,
        std::shared_ptr<MemoryUtils> header_mem_utils,
        std::shared_ptr<MemoryUtils> payload_mem_utils,
        time_handler_ns_cb_t time_handler_cb);
    virtual ~LatencyIONode() = default;
    /**
     * @brief: Prints sender and receiver parameters to an output stream.
     *
     * The method prints the parameters of the sender and the receiver to an output stream.
     *
     * @param [out] out: Output stream to print the parameter to.
     *
     * @return: Reference to the output stream.
     */
    virtual std::ostream& print(std::ostream& out) const = 0;
    /**
     * @brief: Overrides operator << for @ref ral::io_node::LatencyIONode reference.
     */
    friend std::ostream& operator<<(std::ostream& out, const LatencyIONode& sender)
    {
        sender.print(out);
        return out;
    }
    /**
     * @brief: Overrides operator << for @ref ral::io_node::LatencyIONode pointer.
     */
    friend std::ostream& operator<<(std::ostream& out, LatencyIONode* sender)
    {
        sender->print(out);
        return out;
    }
    /**
     * @brief: Initializes send stream (for client requests and server replies).
     */
    virtual void initialize_send_stream() = 0;
    /**
     * @brief: Initializes receive stream (for receiving client requests and server replies).
     */
    virtual void initialize_receive_stream(const TwoTupleFlow& flow) = 0;
    /**
     * @brief: Requests memory sizes required for transmitter and receiver.
     */
    virtual ReturnStatus query_memory_size(size_t& tx_header_size, size_t& tx_payload_size,
                                           size_t& rx_header_size, size_t& rx_payload_size) = 0;
    /**
     * @brief: Distributes the memory for streams.
     *
     * This method is responsible to distribute the memory of the sender/receiver streams.
     */
    virtual void distribute_memory_for_streams(rmx_mem_region& tx_header_mreg,
                                               rmx_mem_region& tx_payload_mreg,
                                               rmx_mem_region& rx_header_mreg,
                                               rmx_mem_region& rx_payload_mreg) = 0;
    /**
     * @brief: Prints sender's parameters.
     *
     * This method is responsible to use the @ref operators << overloaded
     * of sender and it's streams and print the information to standard output.
     *
     * @note: The information will be printed if the sender was initialized with
     *        @ref app_settings->print_parameters parameter of set to true.
     */
    virtual void print_parameters() = 0;
    /**
     * @brief: Client/server loop thread worker.
     *
     * This method is the worker method of the std::thread will run with this object as it's context.
     * The user of @ref ral::io_node::LatencyIONode class can
     * initialize the object in advance and run std::thread when needed.
     */
    virtual void operator()();
    /**
     * @brief: Returns current time in nanoseconds.
     *
     * @returns: Current time in nanoseconds.
     */
    uint64_t get_time_now_ns() { return m_get_time_ns_cb(nullptr);}
protected:
    /**
     * @brief: Creates Rivermax send stream (for client requests and server replies).
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus create_send_stream() = 0;
    /**
     * @brief: Destroys send streams.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus destroy_send_stream() = 0;
    /**
     * @brief: Sets CPU related resources.
     *
     * This method is responsible to set sender's priority and CPU core affinity.
     */
    void set_cpu_resources();
    /**
     * @brief: Prepares the buffers to send.
     *
     * This method is responsible to prepare the data to be sent.
     */
    virtual void prepare_send_buffer() = 0;
    /**
     * @brief: Creates Rivermax Receive stream.
     */
    virtual ReturnStatus create_receive_stream() = 0;
    /**
     * @brief: Destroys Rivermax Receive stream.
     */
    virtual ReturnStatus destroy_receive_stream() = 0;
    /**
     * @brief: Enables receiving unicast/multicast flow with certain destination.
     */
    virtual ReturnStatus attach_receive_flow() = 0;
    /**
     * @brief: Disable receiving earlier enabled unicast/multicast flow.
     */
    virtual ReturnStatus detach_receive_flow() = 0;
    /**
     * @brief: Client-side mearurement loop: sends requests, receives replies, collects and stats.
     */
    virtual void send_receive() = 0;
    /**
     * @brief: Server-side mearurement loop: receives requests, sends replies.
     */
    virtual void receive_send() = 0;
};

} // io_node
} // ral

#endif /* RMAX_APPS_LIB_IO_NODE_MISC_LATENCY_IO_NODE_H_ */
