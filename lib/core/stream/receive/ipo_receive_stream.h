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

#ifndef RMAX_APPS_LIB_LIB_CORE_STREAM_RECEIVE_IPO_RECEIVE_STREAM_H_

#include <chrono>
#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include <unordered_set>

#include <rivermax_api.h>

#include "core/stream/aggregate_stream_interface.h"
#include "core/stream/receive/receive_stream.h"

namespace ral
{
namespace lib
{
namespace core
{

/**
 * @brief: Redundant stream settings.
 */
typedef struct ipo_stream_settings
{
    uint16_t packet_payload_size = 0;
    uint16_t packet_app_header_size = 0;
    size_t num_of_packets_in_chunk = 0;
    size_t max_chunk_size = 0;
    std::unordered_set<rmx_input_option> stream_options;
    uint64_t max_path_differential_us = 0;

    ipo_stream_settings()
    {
    }

    ipo_stream_settings(
            uint16_t packet_payload_size_,
            uint16_t packet_app_header_size_,
            size_t num_of_packets_in_chunk_,
            size_t max_chunk_size_,
            const std::unordered_set<rmx_input_option>& stream_options_,
            uint64_t max_path_differential_us_
    ) :
        packet_payload_size(packet_payload_size_),
        packet_app_header_size(packet_app_header_size_),
        num_of_packets_in_chunk(num_of_packets_in_chunk_),
        max_chunk_size(max_chunk_size_),
        stream_options(stream_options_),
        max_path_differential_us(max_path_differential_us_)
    {
    }

} ipo_stream_settings_t;

/**
 * @brief: This structure describes a receive path (NIC + network flow).
 */
struct IPOReceivePath
{
    std::string dev_ip;
    FourTupleFlow flow;

    IPOReceivePath(const std::string& dev_ip_, const FourTupleFlow& flow_) :
        dev_ip(dev_ip_),
        flow(flow_)
    {
    }
};

/**
 * @brief: This class represents virtual chunks of the aggregated receive stream.
 *         These chunks contain packets produced by @ref IPOReceiveStream
 *         by combining packets from underlying physical redundant receive streams.
 */
class IPOReceiveChunk
{
    bool m_hds_on;
    size_t m_data_block_idx;
    size_t m_chunk_size;
    uint32_t m_seqn_first;
    std::unordered_set<rmx_input_completion_flag> m_comp_flags;
    uint64_t m_timestamp_first;
    uint64_t m_timestamp_last;
    byte_t* m_header_ptr;
    byte_t* m_payload_ptr;
    const ReceivePacketInfo* m_info_ptr;
public:
    IPOReceiveChunk(bool use_hds) :
            m_hds_on(use_hds),
            m_data_block_idx(use_hds ? 1 : 0),
            m_chunk_size(0),
            m_seqn_first(0),
            m_timestamp_first(0),
            m_timestamp_last(0),
            m_header_ptr(nullptr),
            m_payload_ptr(nullptr),
            m_info_ptr(nullptr)
    {}

    ~IPOReceiveChunk() = default;

    /**
     * @brief: Gets the number of packets in the chunk.
     *
     * @return: Number of packets in the chunk.
     */
    size_t get_completion_chunk_size() const { return m_chunk_size; }
    /**
     * @brief: Sets the number of packets in the chunk.
     *
     * @param [in] chunk_size: Number of packets in the chunk.
     */
    void set_completion_chunk_size(size_t chunk_size) { m_chunk_size = chunk_size; }
    /**
     * @brief: Gets the sequence number of the first packet in the chunk.
     *
     * @return: Sequence number of the first packet in the chunk.
     */
    uint32_t get_completion_seqn_first() const { return m_seqn_first; }
    /**
     * @brief: Sets the sequence number of the first packet in the chunk.
     *
     * @param [in] seqn: Sequence number of the first packet in the chunk.
     */
    void set_completion_seqn_first(uint32_t seqn) { m_seqn_first = seqn; }
    /**
     * @brief: Tests the status of the completion flag @p flag.
     *
     * @param [in] flag: Flag type, see @ref rmx_input_completion_flag.
     *
     * @return: True is flag is set.
     */
    bool test_completion_flag(rmx_input_completion_flag flag) const { return m_comp_flags.count(flag) != 0; }
    /**
     * @brief: Sets the status of the completion flag @p flag.
     *
     * @param [in] flag: Flag type, see @ref rmx_input_completion_flag.
     */
    void set_completion_flag(rmx_input_completion_flag flag) { m_comp_flags.insert(flag); }
    /**
     * @brief: Gets the packet header size.
     *
     * @param [in] packet_idx: Packet index in the chunk.
     *
     * @return: Packet header size.
     */
    size_t get_packet_header_size(size_t packet_idx) { return m_hds_on ? m_info_ptr[packet_idx].get_packet_sub_block_size(0) : 0; }
    /**
     * @brief: Gets the packet payload size.
     *
     * @param [in] packet_idx: Packet index in the chunk.
     *
     * @return: Packet payload size.
     */
    size_t get_packet_payload_size(size_t packet_idx) { return m_info_ptr[packet_idx].get_packet_sub_block_size(m_data_block_idx); }
    /**
     * @brief: Gets the timestamp of the specified paket in the chunk.
     *
     * @param [in] packet_idx: Packet index in the chunk.
     *
     * @return: Timestamp of the specified paket in the chunk.
     */
    uint64_t get_packet_timestamp(size_t packet_idx) { return m_info_ptr[packet_idx].get_packet_timestamp(); }
    /**
     * @brief: Gets the first packet timestamp.
     *
     * @return: First packet timestamp.
     */
    uint64_t get_completion_timestamp_first() const { return m_timestamp_first; }
    /**
     * @brief: Sets the first packet timestamp.
     *
     * @param [in] timestamp: First packet timestamp.
     */
    void set_completion_timestamp_first(uint64_t timestamp) { m_timestamp_first = timestamp; }
    /**
     * @brief: Gets the last packet timestamp.
     *
     * @return: Last packet timestamp.
     */
    uint64_t get_completion_timestamp_last() const { return m_timestamp_last; }
    /**
     * @brief: Sets the last packet timestamp.
     *
     * @param [in] timestamp: Last packet timestamp.
     */
    void set_completion_timestamp_last(uint64_t timestamp) { m_timestamp_last = timestamp; }
    /**
     * @brief: Gets the pointer to the underlying header completion structure.
     *
     * @return: Pointer to the underlying header completion structure.
     */
    void* get_completion_header_ptr() const { return m_header_ptr; };
    /**
     * @brief: Sets the pointer to the underlying header completion structure.
     *
     * @param [in] header_ptr: Pointer to the underlying header completion structure.
     */
    void set_completion_header_ptr(byte_t* header_ptr) { m_header_ptr = header_ptr; }
    /**
     * @brief: Gets the pointer to the underlying payload completion structure.
     *
     * @return: Pointer to the underlying payload completion structure.
     */
    void* get_completion_payload_ptr() const { return m_payload_ptr; };
    /**
     * @brief: Sets the pointer to the underlying payload completion structure.
     *
     * @param [in] payload_ptr: Pointer to the underlying payload completion structure.
     */
    void set_completion_payload_ptr(byte_t* payload_ptr) { m_payload_ptr = payload_ptr; }
    /**
     * @brief: Gets the pointer to the array of packet info completion structures.
     *
     * @return: Pointer to the array of packet info completion structures.
     */
    const ReceivePacketInfo* get_completion_info_ptr() {return m_info_ptr; }
    /**
     * @brief: Sets the pointer to the array of packet info completion structures.
     *
     * @param [in] info_ptr: Ppointer to the array of packet info completion structures.
     */
    void set_completion_info_ptr(const ReceivePacketInfo* info_ptr) { m_info_ptr = info_ptr; }
};

/**
 * @brief: Inline Packet Ordering receiver class.
 *
 * This class implements redundant multi-path receive stream by
 * incapsulating multiple streams using Inline Packet Ordering feature over the
 * same memory buffer.
 */
class IPOReceiveStream : public IAggregateStream
{
private:
    using clock = std::chrono::steady_clock;
    static_assert(clock::is_steady, "clock must be steady");

    /**
     * @brief: Extended packet information to be used in stream reconstruction.
     */
    struct ext_packet_info {
        uint32_t sequence_number = 0;
        bool is_valid = false;
        clock::time_point timestamp;
        uint64_t hw_timestamp = 0;
    };

    /**
     * @brief: State of stream reconstruction code.
     */
    enum class State {
        /**
         * Waiting for the first input packet to initialize receiver.
         * Switch to @ref State::Running on a first packet.
         */
        WaitFirstPacket,
        /**
         * Buffer is non-empty. We're waiting for buffered packets processing
         * time. Switching to @ref State::Waiting state once the buffer is empty.
         */
        Running,
        /**
         * Waiting for the next input packet. Transition to @ref
         * State::Running state on packet arrival.
         */
        Waiting
    };

protected:
    const ipo_stream_settings_t m_settings;
    const std::vector<IPOReceivePath> m_paths;
    const bool m_use_ext_seqn;
    const bool m_header_data_split;
    const bool m_pkt_info_enabled;
    const uint32_t m_num_of_packets_in_chunk;

    std::chrono::microseconds m_max_path_differential;

    uint32_t m_sequence_number_wrap_around = 0;
    State m_state = State::WaitFirstPacket;
    std::vector<ReceiveStream> m_streams;
    std::vector<ReceiveChunk> m_chunks;
    byte_t* m_header_buffer = nullptr;
    byte_t* m_payload_buffer = nullptr;
    std::vector<ext_packet_info> m_ext_packet_info_arr;
    std::vector<ReceivePacketInfo> m_packet_info_arr;
    size_t m_header_stride_size = 0;
    size_t m_payload_stride_size = 0;
    size_t m_index = 0;
    clock::time_point m_next_packet_time;
    clock::time_point m_now;
public:
    /**
     * @brief: Constructs Inline Packet Ordering stream.
     *
     * @param [in] id: Stream identifier.
     * @param [in] settings: Stream settings.
     * @param [in] paths: List of redundant data receive paths.
     * @param [in] placement_order: Packet placement order (Rivermax constant).
     */
    IPOReceiveStream(size_t id, const ipo_stream_settings_t& settings,
            const std::vector<IPOReceivePath>& paths, bool use_ext_seqn);

    virtual ~IPOReceiveStream() = default;
    std::ostream& print(std::ostream& out) const override;
    ReturnStatus create_stream() override;
    /**
     * @brief: Attaches flows to all redundant streams.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus attach_flow();
    /**
     * @brief: Detaches flows from all redundant streams.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus detach_flow();
    ReturnStatus destroy_stream() override;
    /**
     * @brief: Receives next chunk from input stream.
     *
     * @param [out] chunk: Pointer to the returned chunk structure.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::signal_received - If operation was interrupted by an OS signal.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    virtual ReturnStatus get_next_chunk(IPOReceiveChunk* chunk);
    /**
     * @brief: Returns the memory size needed by Rivermax to create stream with
     * given parameters.
     *
     * @param [out] header_size: Header buffer size (if header-data split is
     *                          enabled).
     * @param [out] payload_size: Payload buffer size.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus query_buffer_size(size_t& header_size, size_t& payload_size);
    /**
     * @brief: Sets buffers for header and payload data.
     *
     * This method is optional. By default Rivermax will allocate memory
     * buffer(s) internally.
     *
     * @param [in] header_ptr: Pointer to header data buffer. Should be NULL if
     *                         header-data split mode is not enabled.
     * @param [in] payload_ptr: Pointer to payload data buffer.
     */
    void set_buffers(void* header_ptr, void* payload_ptr);
    /**
     * @brief: Sets memory keys for header and payload memory.
     *
     * This method is optional. By default Rivermax will register memory
     * internally.
     *
     * @param [in] header_mkeys: Memory keys for header data buffer. The size of
     *                           vector must be the same as number of paths.
     *                           Ignored if header-data split mode is not
     *                           enabled.
     * @param [in] payload_mkeys: Memory key for payload data buffer. The size of
     *                            vector must be the same as number of paths.
     */
    void set_memory_keys(const std::vector<rmx_mem_region>& header_regions,
            const std::vector<rmx_mem_region>& payload_regions);
    /**
     * @brief: Queries header stride size.
     *
     * @return: Stride size in bytes.
     */
    size_t get_header_stride_size() const { return m_streams[0].get_header_stride_size(); }
    /**
     * @brief: Queries payload stride size.
     *
     * @return: Stride size in bytes.
     */
    size_t get_payload_stride_size() const { return m_streams[0].get_payload_stride_size(); }

protected:
    /**
     * @brief: Handles packet with corrupt RTP header.
     *
     * @param [in] index: Redundant stream index (0-based).
     * @param [in] packet_info: Detailed packet information.
     */
    virtual void handle_corrupted_packet(size_t index, const ReceivePacketInfo& packet_info);
    /**
     * @brief: Handles received packet.
     *
     * This function is called only for the first packet, for redundant packets
     * copies received from another streams @ref handle_redundant_packet will
     * be called.
     *
     * @param [in] index: Redundant stream index (0-based).
     * @param [in] sequence_number: RTP sequence number.
     * @param [in] packet_info: Detailed packet information.
     */
    virtual void handle_packet(size_t index, uint32_t sequence_number, const ReceivePacketInfo& packet_info);
    /**
     * @brief: Handles received redundant packet.
     *
     * This function is called only for redundant packet(s), for the first
     * received packet @ref handle_packet will be called.
     *
     * @param [in] index: Redundant stream index (0-based).
     * @param [in] sequence_number: RTP sequence number.
     * @param [in] packet_info: Detailed packet information.
     */
    virtual void handle_redundant_packet(size_t index, uint32_t sequence_number, const ReceivePacketInfo& packet_info);
    /**
     * @brief: Handles packet before returning it to caller.
     *
     * This function is called when packet is transferred from cache buffer to
     * the caller.
     *
     * @param [in] sequence_number: RTP sequence number.
     */
    virtual void complete_packet(uint32_t sequence_number);
    /**
     * @brief: Returns wrap-around value for sequence number.
     *
     * @return: Wrap-around value for sequence number.
     */
    uint32_t get_sequence_number_wrap_around() const { return m_sequence_number_wrap_around; }
    /**
     * @brief: Parses sequence number from the packet header.
     *
     * @param [in] header: Pointer to the start of the packet header.
     * @param [in] length: Header length.
     * @param [out] sequence_number: Packet sequence number.
     *
     * @return: True on success.
     */
    virtual bool get_sequence_number(const byte_t* header, size_t length, uint32_t& sequence_number) const = 0;
    /**
     * @brief: Gets sequence number mask.
     *
     * @return: Sequence number mask.
     */
    virtual uint32_t get_sequence_number_mask() const = 0;

private:
    /**
     * @brief: Initializes sub-streams for all receive paths.
     */
    void initialize_substreams();
    /**
     * @brief: Handles IO completion returned from substream.
     *
     * @param [in] index: Redundant stream index (0-based).
     * @param [in] stream: Redudnant stream.
     * @param [in] chunk: Received chunk.
     */
    inline void process_completion(size_t index, const ReceiveStream& stream, ReceiveChunk& chunk);
    /**
     * @brief: Returns wrap-round value for sequence number in circular queue of
     * @ref buffer_elements elements.
     *
     * @param [in] buffer_elements: Number of elements in circular buffer.
     *
     * @return: Wrap-around value.
     */
    inline uint32_t get_sequence_number_wrap_around(uint32_t buffer_elements) const;
};

} // namespace core
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_CORE_STREAM_RECEIVE_IPO_RECEIVE_STREAM_H_
#endif /* RMAX_APPS_LIB_LIB_CORE_STREAM_RECEIVE_IPO_RECEIVE_STREAM_H_ */
