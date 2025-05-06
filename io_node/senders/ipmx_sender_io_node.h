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

#ifndef RMAX_APPS_LIB_IO_NODE_SENDERS_IPMX_SENDER_IO_NODE_H_
#define RMAX_APPS_LIB_IO_NODE_SENDERS_IPMX_SENDER_IO_NODE_H_

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>

#include <rivermax_api.h>

#include "api/rmax_apps_lib_api.h"
#include "rivermax_defs.h"
#include "services/error_handling/return_status.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

/*
 * Big-endian mask for version, padding bit and packet type pair
 */
#define RTP_VERSION         2
#define RTCPP_SEND_REPORT   200
#define RTCP_TYPE_VER ((RTP_VERSION << 14) | RTCPP_SEND_REPORT)
#define IPMX_TAG 0x5831
#define IPMX_MIB_TYPE 0x0001

struct IpmxInfoBlock {
    uint16_t ipmx_tag;
    uint16_t length;
    uint8_t block_version;
    uint8_t pad[3];
    uint8_t ts_refclk[60];
    uint8_t mediaclk[12];
};

struct IpmxMediaInfoBlock {
    uint16_t type;
    uint16_t length;
    uint8_t sampling[16];
    uint16_t packing;
    uint8_t par_w;
    uint8_t par_h;
    uint8_t range[12];
    uint8_t colorimetry[20];
    uint8_t tcs[16];
    uint16_t width;
    uint16_t height;
    uint32_t rate;
    uint32_t pixel_clk_hi;
    uint32_t pixel_clk_lo;
    uint16_t htotal;
    uint16_t vtotal;
};

struct IpmxSenderReport {
    uint16_t type_ver;
    uint16_t length;
    uint32_t ssrc;
    uint32_t ntp_ts_hi;
    uint32_t ntp_ts_lo;
    uint32_t rtp_ts;
    uint32_t pkt_cnt;
    uint32_t byte_cnt;
    IpmxInfoBlock info;
    IpmxMediaInfoBlock media;
};

namespace ral
{
namespace io_node
{

/**
 * @brief: A wrapper to Generic send stream to share it between IPMX stream senders.
 *
 * Sends one-packet chunks to specified destinations, marks for completion tracking and
 * extracts sender id and timestamp from transmit completions.
 *
 */

class SharedMessageHandler {
public:
    SharedMessageHandler(const std::shared_ptr<GenericChunk>& chunk_handle);
    ReturnStatus get_next_buffer(rmx_mem_region& mreg);
    ReturnStatus commit_message(const rmx_mem_region& mreg, size_t sender_id, const sockaddr& addr);
    ReturnStatus check_completion(size_t& sender_id, uint64_t& timestamp);
private:
    size_t m_pending_message_count;
    std::shared_ptr<GenericChunk> m_chunk_handle;
};

/**
 * @brief: IPMX stream sender.
 *
 * Implements operations to send one IPMX stream: commits media chunks to keep transmit queue
 * not empty, sends RTCP send reports when needed, using SharedMessageHandler entity.
 *
 */

struct IpmxStreamSender
{
public:
    IpmxStreamSender(
        size_t sender_id,
        const TwoTupleFlow& src_address,
        const TwoTupleFlow& dst_address,
        const media_settings_t& media_settings,
        time_handler_ns_cb_t get_wall_time_ns,
        size_t chunks_in_mem_block,
        size_t packets_in_chunk,
        uint16_t packet_payload_size,
        size_t data_stride_size);
    uint64_t calculate_send_time_ns(uint64_t earliest_start_time_ns);
    void set_initial_timestamps(uint64_t send_start_time, uint64_t report_trigger_time);
    ReturnStatus commit_sender_report();
    ReturnStatus notify_report_completion(uint64_t completion_timestamp);
    ReturnStatus track_media_completions();
    bool has_free_frame_buffer() const;
    bool is_report_for_current_frame_sent() const;
    bool can_sleep(uint64_t& max_wakeup_time) const;
    ReturnStatus commit_next_media_chunk();
    void print_report_stats();
    void init_media_chunk_handler();
    void set_report_chunk_handler(const std::shared_ptr<SharedMessageHandler>& report_handler);
    size_t get_sender_id() const { return m_sender_id; };
    size_t get_stream_number() const { return m_stream_number; };
    RtpVideoSendStream& get_media_stream() { return *m_stream.get(); };
    ReturnStatus start_media_stream();
    ReturnStatus stop_media_stream();
protected:
    ReturnStatus process_media_completion();
    void configure_memory_layout();
    void prepare_sender_report_template();
    size_t m_sender_id;
    size_t m_stream_number;
    std::unique_ptr<RtpVideoSendStream> m_stream;
    std::vector<uint16_t> m_mem_block_payload_sizes;
    std::unique_ptr<MediaChunk> m_media_chunk_handler;
    std::unique_ptr<MediaStreamMemBlockset> m_mem_blockset;
    std::unique_ptr<TwoTupleFlow> m_report_dst_flow;
    std::shared_ptr<SharedMessageHandler> m_report_chunk_handler;
    time_handler_ns_cb_t m_get_wall_time_ns;
    media_settings_t m_media_settings;
    size_t m_chunks_in_mem_block;
    size_t m_packets_in_chunk;
    uint16_t m_packet_payload_size;
    size_t m_data_stride_size;
    uint64_t m_start_send_time_ns;
    uint64_t m_committed_reports;
    uint64_t m_finished_reports;
    uint64_t m_committed_first_chunks;
    uint64_t m_finished_first_chunks;
    uint64_t m_committed_fields;
    uint64_t m_finished_fields;
    size_t m_chunk_in_field_counter;
    uint64_t m_last_report_trigger_ts;
    uint64_t m_last_report_completion_ts;
    bool m_chunk_pending;
    uint64_t m_period_sent_frames_cnt;
    uint64_t m_period_report_delay_sum;
    uint64_t m_period_report_delay_max;
    uint64_t m_period_report_delay_min;
    IpmxSenderReport m_report;
};

/**
 * @brief: IO-node for IPMX senders.
 *
 * Executive container of the IPMX sender that serves as a thread routine
 * with the main-loop of the IPMX streams associated with this node.
 *
 * @note Each sender can run several streams.
 */
class IpmxSenderIONode
{
private:
    /**
    * @brief: Application media send stream rerources
    */

    std::vector<IpmxStreamSender> m_stream_senders;
    std::shared_ptr<GenericSendStream> m_rtcp_stream;
    std::shared_ptr<SharedMessageHandler> m_report_handler;
    std::shared_ptr<GenericChunk> m_rtcp_chunk_handle;
    media_settings_t m_media_settings;
    size_t m_index;
    int m_sleep_between_operations;
    bool m_print_parameters;
    int m_cpu_core_affinity;
    uint16_t m_packet_payload_size;
    size_t m_chunks_in_mem_block;
    size_t m_packets_in_chunk;
    size_t m_data_stride_size;
    size_t m_sender_report_size;
    time_handler_ns_cb_t m_get_nic_time_ns;
    time_handler_ns_cb_t m_get_wall_time_ns;
    rmx_mem_region m_report_mem_region;
    uint64_t m_start_send_time_ns;
public:
    /**
     * @brief: Constructor.
     *
     * @param [in] src_address: Source address of network flow.
     * @param [in] dst_addresses: Destination addresses of network flows.
     * @param [in] app_settings: Application settings.
     * @param [in] index: Index of the sender.
     * @param [in] cpu_core_affinity: CPU core affinity the sender will run on.
     * @param [in] nic_time_hanlder_cb: Time handle callback the IO node uses to get NIC time.
     * @param [in] wall_time_hanlder_cb: Time handle callback the IO node uses to get wall time.
     */
    IpmxSenderIONode(
        const TwoTupleFlow& src_address,
        const std::vector<TwoTupleFlow>& dst_addresses,
        std::shared_ptr<AppSettings>& app_settings,
        size_t index, int cpu_core_affinity,
        time_handler_ns_cb_t nic_time_hanlder_cb,
        time_handler_ns_cb_t wall_time_hanlder_cb);
    virtual ~IpmxSenderIONode() = default;
    /**
     * @brief: Prints sender's parameters to a output stream.
     *
     * The method prints the parameters of the sender to be shown to the user to a output stream.
     *
     * @param [out] out: Output stream parameter print to.
     *
     * @return: Output stream.
     */
    std::ostream& print(std::ostream& out) const;
    /**
     * @brief: Overrides operator << for @ref ral::io_node::IpmxSenderIONode reference.
     */
    friend std::ostream& operator<<(std::ostream& out, const IpmxSenderIONode& sender)
    {
        sender.print(out);
        return out;
    }
    /**
     * @brief: Overrides operator << for @ref ral::io_node::IpmxSenderIONode pointer.
     */
    friend std::ostream& operator<<(std::ostream& out, IpmxSenderIONode* sender)
    {
        sender->print(out);
        return out;
    }
    /**
     * @brief: Initializes stream sender objects.
     */
    void initialize_streams(
        const TwoTupleFlow& src_address,
        const std::vector<TwoTupleFlow>& dst_addresses);
    /**
     * @brief: Requests memory size required for transmitter.
     */
    virtual ReturnStatus query_memory_size(size_t& tx_size) const;
    /**
     * @brief: Initializes sender's memory.
     *
     * Initializes the memory assigned to the sender.
     *
     * @param [in] pointer: Pointer to the allocated memory for the chunks, this memory should be registered first.
     * @param [in] mkey: Memory key from @ref rmx_deregister_memory in Rivermax API.
     *
     * @return: The initialized memory length.
     */
    size_t initialize_memory(void* pointer, rmx_mkey_id mkey);
    /**
     * @brief: Configures memory of Media streams.
     *
     * In this application the memory for media streams is allocated by Rivermax.
     */
    void configure_memory_of_media_streams();
    /**
     * @brief: Prints sender's parameters.
     *
     * This method is responsible to use the @ref operators << overloaded
     * of sender and it's streams and print the information to standard output.
     *
     * @note: The information will be printed if the sender was initialized with
     * @ref app_settings->print_parameters parameter of set to true.
     */
    void print_parameters();
    /**
     * @brief: Sender's worker.
     *
     * This method is the worker method of the std::thread will run with this object as it's context.
     * The user of @ref ral::io_node::IpmxSenderIONode class can
     * initialize the object in advance and run std::thread when needed.
     */
    void operator()();

private:
    /**
     * @brief: Prepares stream senders to start sending data, creates underlying Rivermax objects.
     *
     * This method goes over stream sender objects and
     * creates the appropriate Rivermax stream. It also creates the shared Rivermax Generic
     * send stream used to send RTCP reports.
     *
     * @return: Status of the operation.
     */
    ReturnStatus start_rivermax_streams();
    /**
     * @brief: Terminates data transmit operations and destroys underlying Rivermax objects.
     *
     * This method goes over stream sender objects and
     * destroys the appropriate Rivermax streams. Also destorys the shared Rivermax Generic
     * send stream used to send RTCP reports.
     *
     * @return: Status of the operation.
     */
    ReturnStatus stop_rivermax_streams();
    /**
     * @brief: Sets CPU related resources.
     *
     * Sets sender's priority and CPU core affinity.
     */
    void set_cpu_resources();
    /**
     * @brief: Prepares the buffers to send.
     *
     * Prepares the data to be sent by stream senders.
     * It should be called after @ref ral::io_node::IpmxSenderIONode::initialize_streams.
     */
    inline void prepare_buffers();
    /**
     * @brief: Returns current NIC time in nanoseconds.
     *
     * @returns: Current time in nanoseconds.
     */
    uint64_t get_nic_time_now_ns() { return m_get_nic_time_ns(nullptr); }
    /**
     * @brief: Returns current wall time in nanoseconds.
     *
     * @returns: Current time in nanoseconds.
     */
    uint64_t get_wall_time_now_ns() { return m_get_wall_time_ns(nullptr); }
    /**
     * @brief: Waits until the specified time.
     *
     * Waits until the specified time in nanoseconds. Returns before this time.
     *
     * @param [in] return_time_ns: Time when to return, in nanoseconds.
     */
    inline void wait_until(uint64_t return_time_ns);
    /**
     * @brief: Sensd IPMX Send Reports for all streams before sending data.
     *
     * @return: Status of the operation.
     */
    ReturnStatus send_initial_reports();
    /**
     * @brief: Polls for Send Report send completions.
     *
     * Tries to fetch report completions: all media streams share one generic send stream
     * for sending reports.
     *
     * @return: Status of the operation, if no completions returns ReturnStatus::success
     */
    ReturnStatus check_report_completion();
};

} // io_node
} // ral

#endif /* RMAX_APPS_LIB_IO_NODE_SENDERS_IPMX_SENDER_IO_NODE_H_ */
