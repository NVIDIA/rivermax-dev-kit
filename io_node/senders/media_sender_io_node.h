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

#ifndef RMAX_APPS_LIB_IO_NODE_SENDERS_MEIDA_SENDER_IO_NODE_H_

#include <cstddef>
#include <vector>
#include <unordered_map>
#include <memory>
#include <iostream>
#include <ostream>
#include <chrono>

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
constexpr uint8_t LEAP_SECONDS = 37;
constexpr uint16_t FHD_WIDTH = 1920;
constexpr uint16_t FHD_HEIGHT = 1080;
constexpr uint16_t UHD_HEIGHT = 2160;
constexpr uint16_t UHD_WIDTH = 3840;
constexpr uint8_t VIDEO_TRO_DEFAULT_MODIFICATION = 2;
constexpr size_t SLEEP_THRESHOLD_NS = std::chrono::nanoseconds{ std::chrono::milliseconds{ 2 } }.count();
constexpr size_t NS_IN_SEC = std::chrono::nanoseconds{ std::chrono::seconds{ 1 } }.count();
/**
 * @brief: Sending statistics struct.
 *
 * This struct will hold run time statistics of a stream.
 */
struct SendingStatistics
{
    uint32_t packet_counter;
    uint32_t rtp_sequence;
    uint32_t rtp_timestamp;
    uint8_t rtp_interlace_field_indicator;
    uint16_t line_number;
    uint16_t srd_offset;
};
/**
 * @brief: Application Media API send stream interface.
 *
 * This class implements and extends @ref ral::lib::core::MediaSendStream operations.
 */
class AppMediaSendStream : public MediaSendStream
{
private:
    SendingStatistics m_send_stats;
    const media_settings_t m_media_settings;
public:
    /**
     * @brief: AppMediaSendStream constructor.
     *
     * @param [in] settings: Stream parameters.
     * @param [in] mem_blocks: Parameters of blocks allocated for output packets.
     */
    AppMediaSendStream(const MediaStreamSettings& settings, MediaStreamMemBlockset& mem_blocks);
    virtual ~AppMediaSendStream() = default;
    std::ostream& print(std::ostream& out) const override;
    /**
     * @brief: Prepares media chunk to send.
     *
     * This method will prepare RTP header for the given chunk.
     * The media payload will be random payload.
     *
     * @param [in] chunk: Media chunk.
     *
     * @note: TODO: Remove this and add the appropriate buffer writer interface in the library for this.
     */
    inline void prepare_chunk_to_send(MediaChunk& chunk);
    /**
     * @brief: Calculates stream send time.
     *
     * This method will calculate the time to send the stream based on SPMTE 2110, from the given time plus 1 second.
     * It will also update with the appropriate rtp_timestamp in the sending statistics of the stream.
     *
     * @param [in] time_now_ns: Send time in nanosecond starting point.
     *
     * @note: TODO: Remove this and add send time calculation logic components in the library.
     *
     * @return: Time to send the stream in nanoseconds.
     */
    double calculate_send_time_ns(uint64_t time_now_ns);
private:
    /**
     * @brief: Builds 2110-20 RTP header.
     *
     * @param [in] buffer: Pointer to the buffer of the packet to fill.
     *
     * @note: TODO: Remove this and add stream buffer writer for RTP components in the library.
     */
    inline void build_2110_20_rtp_header(byte_t* buffer);
};

/**
 * @brief: MediaSenderIONode class.
 *
 * This class implements the required operations in order
 * to be a sender. The sender class will be the context
 * that will be run under a std::thread by overriding the operator ().
 * Each sender will be able to run multiple streams.
 */
class MediaSenderIONode
{
private:
    /**
    * @brief: Application media send stream rerources
    */
    struct MediaStreamPack
    {
        std::unique_ptr<AppMediaSendStream> stream;
        std::unique_ptr<MediaChunk> chunk_handler;
        std::unique_ptr<MediaStreamMemBlockset> mem_blockset;
        std::vector<TwoTupleFlow> flows;
    };
    std::vector<MediaStreamPack> m_stream_packs;
    media_settings_t m_media_settings;
    size_t m_index;
    FourTupleFlow m_network_address;
    int m_sleep_between_operations;
    bool m_print_parameters;
    rmax_cpu_set_t m_cpu_affinity_mask;
    int m_cpu_core_affinity;
    uint32_t m_hw_queue_full_sleep_us;
    std::unique_ptr<IBufferWriter> m_buffer_writer;
    std::shared_ptr<MemoryUtils> m_mem_utils;
    size_t m_num_of_chunks_in_mem_block;
    uint16_t m_packet_payload_size;
    size_t m_num_of_packets_in_chunk;
    size_t m_num_of_packets_in_mem_block;
    size_t m_data_stride_size;
    std::vector<uint16_t> m_mem_block_payload_sizes;
    uint8_t m_dscp, m_pcp, m_ecn;
    time_handler_ns_cb_t m_get_time_ns_cb;
public:
    /**
     * @brief: MediaSenderIONode constructor.
     *
     * @param [in] network_address: Network address of the IO node.
     * @param [in] app_settings: Application settings.
     * @param [in] index: Index of the sender.
     * @param [in] num_of_streams: Number of streams in the sender.
     * @param [in] cpu_core_affinity: CPU core affinity the sender will run on.
     * @param [in] mem_utils: Memory utilities.
     * @param [in] time_hanlder_cb: Time handle callback the IO node will use to get current time.
     */
    MediaSenderIONode(
        const FourTupleFlow& network_address,
        std::shared_ptr<AppSettings> app_settings,
        size_t index, size_t num_of_streams, int cpu_core_affinity,
        std::shared_ptr<MemoryUtils> mem_utils, time_handler_ns_cb_t time_hanlder_cb);
    virtual ~MediaSenderIONode() = default;
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
     * @brief: Overrides operator << for @ref ral::io_node::MediaSenderIONode reference.
     */
    friend std::ostream& operator<<(std::ostream& out, const MediaSenderIONode& sender)
    {
        sender.print(out);
        return out;
    }
    /**
     * @brief: Overrides operator << for @ref ral::io_node::MediaSenderIONode pointer.
     */
    friend std::ostream& operator<<(std::ostream& out, MediaSenderIONode* sender)
    {
        sender->print(out);
        return out;
    }
    /**
     * @brief: Initializes send flows objects.
     *
     * This method will initialize the send flows uniformly among it's streams.
     *
     * @param [in] flows: Flows assigned to sender's streams.
     */
    void initialize_send_flows(const std::vector<TwoTupleFlow>& flows);
    /**
     * @brief: Initializes stream objects.
     */
    void initialize_streams();
    /**
     * @brief: Initializes sender's memory.
     *
     * This method is responsible to take the memory assigned to the sender
     * and distribute it to the streams of this sender
     * using @ref ral::io_node::MediaSenderIONode::distribute_memory_for_streams.
     */
    void initialize_memory();
    /**
     * @brief: Distributes the memory for streams.
     *
     * This method is responsible to distribute the memory of the sender
     * to it's streams.
     */
    void distribute_memory_for_streams();
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
     * The user of @ref ral::io_node::MediaSenderIONode class can
     * initialize the object in advance and run std::thread when needed.
     */
    void operator()();
private:
    /**
     * @brief: Creates sender's streams.
     *
     * This method is responsible to go over sender's stream objects and
     * create the appropriate Rivermax stream.
     *
     * @return: Status of the operation.
     */
    ReturnStatus create_streams();
    /**
     * @brief: Destroys sender's streams.
     *
     * This method is responsible to go over sender's stream objects and
     * destroy the appropriate Rivermax stream.
     *
     * @return: Status of the operation.
     */
    ReturnStatus destroy_streams();
    /**
     * @brief: Sets CPU related resources.
     *
     * This method is responsible to set sender's priority and CPU core affinity.
     */
    void set_cpu_resources();
    /**
     * @brief: Prepares the buffers to send.
     *
     * This method is responsible to prepare the data to be sent for it's streams.
     * It should be called after @ref ral::io_node::MediaSenderIONode::initialize_streams.
     */
    inline void prepare_buffers();
    /**
     * @brief: Returns current time in nanoseconds.
     *
     * @returns: Current time in nanoseconds.
     */
    uint64_t get_time_now_ns() { return m_get_time_ns_cb(nullptr); }
    /**
     * @brief: Waits for the next frame.
     *
     * This method implements logic to wait and wake up when next frame send time is close.
     *
     * @param [in] send_time_ns: Send time of the next frame in nanoseconds.
     */
    inline void wait_for_next_frame(uint64_t send_time_ns);
};

} // io_node
} // ral

#define RMAX_APPS_LIB_IO_NODE_SENDERS_MEIDA_SENDER_IO_NODE_H_
#endif /* RMAX_APPS_LIB_IO_NODE_SENDERS_MEIDA_SENDER_IO_NODE_H_ */
