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
 * @brief: This structure describes receive path (NIC + network flow).
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
 * @brief: Inline Packet Ordering receiver class.
 *
 * This class implements redundant multi-path receive stream by
 * incapsulating multiple streams using Inline Packet Ordering feature over the
 * same memory buffer.
 */
class IPOReceiveStream : public IAggregateStream
{
protected:
    const std::vector<IPOReceivePath> m_paths;

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

    const bool m_pkt_info_enabled;
    const bool m_header_data_split;
    const uint32_t m_num_of_packets_in_chunk;

    std::chrono::microseconds m_max_path_differential;

    uint32_t m_sequence_number_wrap_around = 0;
    State m_state = State::WaitFirstPacket;
    receive_stream_settings_t m_stream_settings;
    std::vector<ReceiveStream> m_streams;
    byte_ptr_t m_header_buffer = nullptr;
    byte_ptr_t m_payload_buffer = nullptr;
    std::vector<ext_packet_info> m_ext_packet_info_arr;
    std::vector<rmax_in_packet_info> m_packet_info_arr;
    uint16_t m_header_stride_size = 0;
    uint16_t m_payload_stride_size = 0;
    size_t m_index = 0;
    clock::time_point m_next_packet_time;
    clock::time_point m_now;

public:
    /**
     * @brief: Construct Inline Packet Ordering stream.
     *
     * @param [in] id: Stream identifier.
     * @param [in] settings: Stream settings.
     * @param [in] paths: List of redundant data receive paths.
     * @param [in] placement_order: Packet placement order (Rivermax constant).
     */
    IPOReceiveStream(size_t id, const ipo_stream_settings_t& settings,
            const std::vector<IPOReceivePath>& paths,
            rmax_in_buffer_attr_flags_t placement_order);
    virtual ~IPOReceiveStream() = default;
    /**
     * @brief: Print detailed stream information.
     *
     * @param [out] out: Output stream to print information.
     *
     * @return: Output stream.
     */
    std::ostream& print(std::ostream& out) const override;
    /**
     * @brief: Creates redundant streams.
     *
     * This method is responsible to create receive streams for all paths.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus create_stream() override;
    /**
     * @brief: Attach flows to all redundant streams.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus attach_flow();
    /**
     * @brief: Detach flows from all redundant streams.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus detach_flow();
    /**
     * @brief: Destroy Redundant streams.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    ReturnStatus destroy_stream() override;
    /**
     * @brief: Receive next chunk from input stream.
     *
     * @param [out] chunk: Pointer to the returned chunk structure.
     *
     * @return: Status of the operation:
     *          @ref ral::lib::services::ReturnStatus::success - In case of success.
     *          @ref ral::lib::services::ReturnStatus::signal_received - If operation was interrupted by an OS signal.
     *          @ref ral::lib::services::ReturnStatus::failure - In case of failure, Rivermax status will be logged.
     */
    virtual ReturnStatus get_next_chunk(ReceiveChunk* chunk);
    /**
     * @brief: Return the memory size needed by Rivermax to create stream with
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
     * @brief: Set buffers for header and payload data.
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
     * @brief: Query header stride size.
     *
     * @return: Stride size in bytes.
     */
    uint16_t get_header_stride_size() const { return m_streams[0].get_header_stride_size(); }
    /**
     * @brief: Query payload stride size.
     *
     * @return: Stride size in bytes.
     */
    uint16_t get_payload_stride_size() const { return m_streams[0].get_payload_stride_size(); }

protected:
    /**
     * @brief: Handle packet with corrupt RTP header.
     *
     * @param [in] index: Redundant stream index (0-based).
     * @param [in] info: Detailed packet information.
     */
    virtual void handle_corrupted_packet(size_t index, const rmax_in_packet_info& info);
    /**
     * @brief: Handle received packet.
     *
     * This function is called only for the first packet, for redundant packets
     * copies received from another streams @ref handle_redundant_packet will
     * be called.
     *
     * @param [in] index: Redundant stream index (0-based).
     * @param [in] sequence_number: RTP sequence number.
     * @param [in] info: Detailed packet information.
     */
    virtual void handle_packet(size_t index, uint32_t sequence_number, const rmax_in_packet_info& info);
    /**
     * @brief: Handle received redundant packet.
     *
     * This function is called only for redundant packet(s), for the first
     * received packet @ref handle_packet will be called.
     *
     * @param [in] index: Redundant stream index (0-based).
     * @param [in] sequence_number: RTP sequence number.
     * @param [in] info: Detailed packet information.
     */
    virtual void handle_redundant_packet(size_t index, uint32_t sequence_number, const rmax_in_packet_info& info);
    /**
     * @brief: Handle packet before returning it to caller.
     *
     * This function is called when packet is transferred from cache buffer to
     * the caller.
     *
     * @param [in] sequence_number: RTP sequence number.
     */
    virtual void complete_packet(uint32_t sequence_number);
    /**
     * @brief: Return wrap-around value for sequence number.
     *
     * @return: Wrap-around value for sequence number.
     */
    uint32_t get_sequence_number_wrap_around() const { return m_sequence_number_wrap_around; }
    /**
     * @brief: Parse sequence number from the packet header.
     *
     * @param [in] header: Pointer to the start of the packet header.
     * @param [in] length: Header length.
     * @param [out] sequence_number: Packet sequence number.
     *
     * @return: True on success.
     */
    virtual bool get_sequence_number(const byte_ptr_t header, size_t length, uint32_t& sequence_number) const = 0;
    /**
     * @brief: Get sequence number mask.
     *
     * @return: Sequence number mask.
     */
    virtual uint32_t get_sequence_number_mask() const = 0;

private:
    /**
     * @brief: Initialize sub-streams for all receive paths.
     */
    void initialize_substreams();
    /**
     * @brief: Handle IO completion returned from substream.
     *
     * @param [in] index: Redundant stream index (0-based).
     * @param [in] stream: Redudnant stream.
     * @param [in] chunk: Pointer to received chunk.
     */
    inline void process_completion(size_t index, const ReceiveStream& stream, ReceiveChunk *chunk);
    /**
     * @brief: Return wrap-round value for sequence number in circular queue of
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
