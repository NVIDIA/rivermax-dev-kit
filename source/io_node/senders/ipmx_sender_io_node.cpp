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

#include <cstdint>
#include <thread>
#include <cstddef>
#include <vector>
#include <memory>
#include <iostream>
#include <ostream>
#include <cstring>
#include <regex>

#include <rivermax_api.h>
#include "rt_threads.h"

#include "rdk/io_node/senders/ipmx_sender_io_node.h"
#include "rdk/services/error_handling/error_handling.h"
#include "rdk/services/cpu/affinity/affinity.h"
#include "rdk/services/utils/defs.h"
#include "rdk/services/media/media.h"
#include "rdk/services/utils/enum_utils.h"
#include "rdk/core/chunk/generic_chunk.h"
#include "rdk/core/flow/flow.h"

using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

static constexpr uint64_t LAST_CHUNK_IN_FRAME_TOKEN = 0xffffffffffffffff;

/**
* @brief: Modifies a given field in the SDP file.
*
* @pram [in] sdp_contents: SDP file to change.
* @pram [in] prefix_str: Prefix string before source_str.
* @pram [in] new_str: The new string to replace.
* @pram [in] suffix_str: Suffix string after source_str.
*/
static inline void modify_sdp_field(
    std::string& sdp_contents, const std::string& prefix,
    const std::string& replacement, const char* suffix)
{
    std::regex re(prefix + ".*?" + suffix);
    sdp_contents = std::regex_replace(sdp_contents, re, prefix + replacement + suffix);
}

SharedMessageHandler::SharedMessageHandler(const std::shared_ptr<GenericChunk>& chunk_handle) :
    m_pending_message_count(0),
    m_chunk_handle(chunk_handle)
{}

ReturnStatus SharedMessageHandler::get_next_buffer(rmx_mem_region& mreg)
{
    ReturnStatus status = m_chunk_handle->get_next_chunk();
    if (status != ReturnStatus::success) {
        return status;
    }

    auto& packet = m_chunk_handle->get_packet(0);
    mreg = packet[0];
    return ReturnStatus::success;
}

ReturnStatus SharedMessageHandler::commit_message(const rmx_mem_region& mreg, size_t sender_id, const sockaddr& addr)
{
    ReturnStatus status;
    m_chunk_handle->set_dest_address(addr);
    auto& packet = m_chunk_handle->get_packet(0);
    packet[0] = mreg;
    status = m_chunk_handle->apply_packets_layout();
    if (status != ReturnStatus::success) {
        if (status != ReturnStatus::signal_received) {
            std::cerr << "Error preparing a Sender Report chunk to send..." << std::endl;
        }
        return status;
    }
    m_chunk_handle->mark_for_tracking(sender_id);
    m_chunk_handle->commit_chunk(0);
    m_pending_message_count++;
    return ReturnStatus::success;
}

ReturnStatus SharedMessageHandler::check_completion(size_t& sender_id, uint64_t& timestamp)
{
    if (m_pending_message_count == 0) {
        return ReturnStatus::no_completion;
    }
    ReturnStatus status = m_chunk_handle->poll_for_completion();
    if (status == ReturnStatus::no_completion) {
        return status;
    }
    if (status != ReturnStatus::success) {
        if (status != ReturnStatus::signal_received) {
            std::cerr << "Failed polling for Report Tx completion" << std::endl;
        }
        return status;
    }
    uint64_t tx_hw_timestamp;
    uint64_t compl_token;
    status = m_chunk_handle->get_last_completion_info(tx_hw_timestamp, compl_token);
    if (status != ReturnStatus::success) {
        std::cerr << "Failed to get completion info for a Send Report" << std::endl;
        return status;
    }
    m_pending_message_count--;
    sender_id = static_cast<size_t>(compl_token);
    timestamp = tx_hw_timestamp;
    return ReturnStatus::success;
}

IPMXStreamSender::IPMXStreamSender(size_t sender_id, const TwoTupleFlow& src_address,
        const TwoTupleFlow& dst_address, const MediaSettings& media_settings,
        time_handler_ns_cb_t get_wall_time_ns, size_t chunks_in_mem_block,
        size_t packets_in_chunk, uint16_t packet_payload_size, size_t data_stride_size) :
    m_sender_id(sender_id),
    m_stream_number{dst_address.get_id()},
    m_get_wall_time_ns(std::move(get_wall_time_ns)),
    m_media_settings(media_settings),
    m_chunks_in_mem_block{chunks_in_mem_block},
    m_packets_in_chunk{packets_in_chunk},
    m_packet_payload_size{packet_payload_size},
    m_data_stride_size{data_stride_size},
    m_start_send_time_ns{0},
    m_committed_reports{0},
    m_finished_reports{0},
    m_committed_first_chunks{0},
    m_finished_first_chunks{0},
    m_committed_fields{0},
    m_finished_fields{0},
    m_chunk_in_field_counter{0},
    m_last_report_trigger_ts{0},
    m_last_report_completion_ts{0},
    m_chunk_pending{false},
    m_period_sent_frames_cnt{0},
    m_period_report_delay_sum{0},
    m_period_report_delay_max{0},
    m_period_report_delay_min{0}
{
    auto& dst_ip = dst_address.get_ip();
    auto dst_port = dst_address.get_port();

    m_report_dst_flow = std::make_unique<TwoTupleFlow>(m_stream_number, dst_ip, dst_port + 1);
    modify_sdp_field(m_media_settings.sdp, "c=IN IP4 ", dst_ip, "/");
    modify_sdp_field(m_media_settings.sdp, "incl IN IP4 ", dst_ip, " ");
    modify_sdp_field(m_media_settings.sdp, "m=video ", std::to_string(dst_port), " ");

    MediaStreamSettings stream_settings(src_address, m_media_settings,
                m_packets_in_chunk, m_packet_payload_size, m_data_stride_size);

    configure_memory_layout();
    m_stream = std::make_unique<RtpVideoSendStream>(stream_settings, *m_mem_blockset.get());

    prepare_compound_report_template(src_address);
}

void IPMXStreamSender::configure_memory_layout()
{
    m_mem_block_payload_sizes.resize(m_chunks_in_mem_block * m_packets_in_chunk, m_packet_payload_size);

    m_mem_blockset = std::make_unique<MediaStreamMemBlockset>(
            1, 1, m_chunks_in_mem_block);
    m_mem_blockset->set_rivermax_to_allocate_memory();
    m_mem_blockset->set_block_layout(0, m_mem_block_payload_sizes.data(), nullptr);
}

size_t IPMXStreamSender::prepare_sender_report_sdes(uint32_t ssrc, const TwoTupleFlow& src_address)
{
    uint8_t* ptr = m_report.sdes_raw;
    RTCPSourceDescr* sdes = new (ptr) RTCPSourceDescr;
    ptr += sizeof(RTCPSourceDescr);
    sdes->type_ver = htons((RTP_VERSION << 14) | (RTCP_SOURCE_CNT << 8) |
                           static_cast<uint16_t>(RTCPPacketType::SourceDescr));
    RTCPSdesChunk* chunk = new (ptr) RTCPSdesChunk;
    ptr += sizeof(RTCPSdesChunk);
    std::string cname = "ipmx@" + src_address.get_ip();
    chunk->ssrc = htonl(ssrc);
    RTCPSdesCname* sdes_cname_item = new (ptr) RTCPSdesCname;
    ptr += sizeof(RTCPSdesCname);
    sdes_cname_item->type = static_cast<uint8_t>(RTCPSdesType::Cname);
    sdes_cname_item->length = static_cast<uint8_t>(cname.length());
    size_t sdes_cname_item_size = sizeof(RTCPSdesCname) + sdes_cname_item->length;
    std::strncpy((char *)(ptr), cname.c_str(),
                 RTCP_SDES_CNAME_LEN_MAX);
    ptr += sdes_cname_item->length;
    RTCPSdesEnd* sdes_end_item = new (ptr) RTCPSdesEnd;
    ptr += sizeof(RTCPSdesEnd);
    sdes_end_item->type = static_cast<uint8_t>(RTCPSdesType::End);
    size_t sdes_size = align_up_pow2(ptr - m_report.sdes_raw, sizeof(uint32_t));
    sdes->length = htons(static_cast<uint16_t>(sdes_size) / sizeof(uint32_t) - 1);
    return sdes_size;
}

size_t IPMXStreamSender::prepare_sender_report_base(uint32_t ssrc, const TwoTupleFlow& src_address)
{
    m_report.sr.type_ver = htons((RTP_VERSION << 14) |
                                 static_cast<uint16_t>(RTCPPacketType::SenderReport));
    m_report.sr.ssrc = htonl(ssrc);
    m_report.sr.length = htons(sizeof(m_report.sr) / sizeof(uint32_t) - 1);
    m_report.sr.info.ipmx_tag = htons(IPMX_TAG);
    m_report.sr.info.length = htons(sizeof(m_report.sr.info) / sizeof(uint32_t) - 1);
    std::strncpy((char *)(m_report.sr.info.ts_refclk), m_media_settings.refclk.c_str(),
                          sizeof(m_report.sr.info.ts_refclk) - 1);
    std::strncpy((char *)(m_report.sr.info.mediaclk), "direct=0",
                          sizeof(m_report.sr.info.mediaclk) - 1);

    m_report.sr.media.type = htons(IPMX_MIB_TYPE_UNCOMPRESSED_VIDEO);
    m_report.sr.media.length = htons(sizeof(m_report.sr.media) / sizeof(uint32_t) - 1);
    std::strncpy((char *)(m_report.sr.media.sampling),
            enum_to_string(m_media_settings.sampling_type).c_str(),
            sizeof(m_report.sr.media.sampling) - 1);
    m_report.sr.media.packing = htons(pack_media_bits_interlace(std::stoi(enum_to_string(m_media_settings.bit_depth)),
            (m_media_settings.video_scan_type == VideoScanType::Interlaced)));
    std::strncpy((char *)(m_report.sr.media.range), "NARROW", sizeof(m_report.sr.media.range) - 1);
    std::strncpy((char *)(m_report.sr.media.colorimetry), "BT709",
            sizeof(m_report.sr.media.colorimetry) - 1);
    std::strncpy((char *)(m_report.sr.media.tcs), "SDR", sizeof(m_report.sr.media.tcs) - 1);
    m_report.sr.media.par_h = 1;
    m_report.sr.media.par_w = 1;
    m_report.sr.media.width = htons(m_media_settings.resolution.width);
    m_report.sr.media.height = htons(m_media_settings.resolution.height);
    uint64_t pixel_clock = ((static_cast<uint64_t>(m_media_settings.resolution.width *
                                                   m_media_settings.resolution.height) *
                            m_media_settings.frame_rate.num) / m_media_settings.frame_rate.denom);
    m_report.sr.media.pixel_clk_hi = htonl(static_cast<uint32_t>(pixel_clock / NS_IN_SEC));
    m_report.sr.media.pixel_clk_lo = htonl(static_cast<uint32_t>(pixel_clock % NS_IN_SEC));
    m_report.sr.media.htotal = m_report.sr.media.width;
    m_report.sr.media.vtotal = m_report.sr.media.height;
    m_report.sr.media.rate = htonl((m_media_settings.frame_rate.num) << 10 |
                                   (m_media_settings.frame_rate.denom));
    return sizeof(IPMXSenderReport);
}

void IPMXStreamSender::prepare_compound_report_template(const TwoTupleFlow& src_address)
{
    uint32_t ssrc = static_cast<uint32_t>(m_stream_number);
    memset(&m_report, 0, sizeof(m_report));

    size_t base_size = prepare_sender_report_base(ssrc, src_address);
    size_t sdes_size = prepare_sender_report_sdes(ssrc, src_address);

    m_report_size = base_size + sdes_size;
}

ReturnStatus IPMXStreamSender::notify_report_completion(uint64_t completion_timestamp)
{
    m_last_report_completion_ts = completion_timestamp;
    uint64_t report_delay = m_last_report_completion_ts - m_last_report_trigger_ts;

#ifdef SEND_REPORT_DEBUG
    std::cout << "SR" << m_id << " delay " << report_delay << std:: endl;
#endif
    if (m_committed_reports <= m_finished_reports) {
        std::cerr << "Unsolicited Send Report completion" << std::endl;
        return ReturnStatus::failure;
    }
    m_finished_reports++;
    m_period_report_delay_sum += report_delay;
    if (m_period_sent_frames_cnt == 0) {
        m_period_report_delay_max = report_delay;
        m_period_report_delay_min = report_delay;
    } else {
        m_period_report_delay_max = (report_delay > m_period_report_delay_max) ?
                                    report_delay : m_period_report_delay_max;
        m_period_report_delay_min = (report_delay < m_period_report_delay_min) ?
                                    report_delay : m_period_report_delay_min;
    }
    m_period_sent_frames_cnt++;
    return ReturnStatus::success;
}

uint64_t IPMXStreamSender::calculate_send_time_ns(uint64_t earliest_start_time_ns)
{
    return m_stream->calculate_send_time_ns(earliest_start_time_ns);
}

void IPMXStreamSender::set_initial_timestamps(uint64_t send_start_time, uint64_t report_trigger_time)
{
    m_start_send_time_ns = send_start_time;
    m_last_report_trigger_ts = report_trigger_time;
}

ReturnStatus IPMXStreamSender::commit_sender_report()
{
    rmx_mem_region mreg;

    ReturnStatus status = m_report_chunk_handler->get_next_buffer(mreg);
    if (status != ReturnStatus::success) {
        return status;
    }

    uint64_t timestamp = m_start_send_time_ns + m_media_settings.frame_field_time_interval_ns *
                                         m_finished_first_chunks;
    m_report.sr.ntp_ts_hi = htonl(static_cast<uint32_t>(timestamp / NS_IN_SEC));
    m_report.sr.ntp_ts_lo = htonl(static_cast<uint32_t>(timestamp % NS_IN_SEC));
    uint32_t rtp_ts = static_cast<uint32_t>(
            time_to_rtp_timestamp(timestamp, static_cast<int>(m_media_settings.sample_rate)));
    m_report.sr.rtp_ts = htonl(rtp_ts);
    uint32_t sent_pkt_cnt = m_finished_fields * m_media_settings.packets_in_frame_field;
    m_report.sr.pkt_cnt = htonl(sent_pkt_cnt);
    uint32_t payload_octets_in_packet = m_packet_payload_size -
                                        (RTP_HEADER_SIZE + RTP_SINGLE_SRD_HEADER_SIZE);
    m_report.sr.byte_cnt = htonl(sent_pkt_cnt * payload_octets_in_packet);

    auto* report = new (mreg.addr) RTCPCompoundPacket;
    *report = m_report;
    mreg.length = m_report_size;

    status = m_report_chunk_handler->commit_message(mreg, m_sender_id, m_report_dst_flow->get_socket_address());

    if (status != ReturnStatus::success) {
        if (status != ReturnStatus::signal_received) {
            std::cerr << "Failed to commit a Sender Report" << std::endl;
        }
        return status;
    }
    m_committed_reports++;
    return ReturnStatus::success;
}

ReturnStatus IPMXStreamSender::track_media_completions()
{
    ReturnStatus status = ReturnStatus::success;

    status = m_media_chunk_handler->poll_for_completion();
    switch (status) {
    case ReturnStatus::no_completion:
        status = ReturnStatus::success;
        break;
    case ReturnStatus::signal_received:
        break;
    case ReturnStatus::success:
        status = process_media_completion();
        break;
    default:
        std::cerr << "Failed polling for Tx completion" << std::endl;
    }
    return status;
}

ReturnStatus IPMXStreamSender::process_media_completion()
{
    ReturnStatus status = ReturnStatus::success;
    uint64_t tx_hw_timestamp;
    uint64_t compl_token;
    status = m_media_chunk_handler->get_last_completion_info(tx_hw_timestamp, compl_token);
    if (status != ReturnStatus::success) {
        std::cerr << "Failed to get completion info for a sent chunk" << std::endl;
        return status;
    }
    if (compl_token == LAST_CHUNK_IN_FRAME_TOKEN) {
        m_finished_fields++;
        return ReturnStatus::success;
    }
    if (compl_token != m_finished_first_chunks) {
        std::cerr << "Out-of-order Tx completion (expected "
                  << m_finished_first_chunks << " got " << compl_token
                  << ")" << std::endl;
        status = ReturnStatus::failure;
        return status;
    }

    m_last_report_trigger_ts = tx_hw_timestamp;
    m_finished_first_chunks++;

    if (m_committed_reports > m_finished_reports) {
        std::cerr << "Send report timeout" << std::endl;
    }

    status = commit_sender_report();
    if (status != ReturnStatus::success) {
        if (status != ReturnStatus::signal_received) {
            std::cerr << "Failed to send an IPMX Sender Report" << std::endl;
        }
        return status;
    }

    return ReturnStatus::success;
}

bool IPMXStreamSender::has_free_frame_buffer() const
{
    return m_committed_fields < m_finished_fields + m_media_settings.frames_fields_in_mem_block;
}

bool IPMXStreamSender::is_report_for_current_frame_sent() const
{
    return m_finished_reports >= m_finished_fields + 1;
}

bool IPMXStreamSender::can_sleep(uint64_t& max_wakeup_time) const
{
    if (has_free_frame_buffer() || !is_report_for_current_frame_sent()) {
        return false;
    }
    max_wakeup_time = m_start_send_time_ns + m_media_settings.frame_field_time_interval_ns *
                                             m_finished_first_chunks;
    return true;
}

ReturnStatus IPMXStreamSender::commit_next_media_chunk()
{
    ReturnStatus status = ReturnStatus::success;

    if (!m_chunk_pending) {
        status = m_stream->get_next_chunk(*m_media_chunk_handler);
        if (unlikely(status != ReturnStatus::success)) {
            if (status != ReturnStatus::signal_received) {
                std::cerr << "Failed to get a next chunk to send" << std::endl;
            }
            return status;
        }
        m_stream->prepare_chunk_to_send(*m_media_chunk_handler);
    } else {
        m_chunk_pending = false;
    }

    uint64_t commit_timestamp_ns = 0;
    if (m_chunk_in_field_counter == 0) {

        status = m_media_chunk_handler->mark_for_tracking(m_committed_first_chunks);
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to mark the first chunk for tracking" << std::endl;
            return status;
        }

        commit_timestamp_ns = m_start_send_time_ns +
                              m_media_settings.frame_field_time_interval_ns *
                              m_committed_fields;

    } else if (m_chunk_in_field_counter == m_media_settings.chunks_in_frame_field - 1) {
        status = m_media_chunk_handler->mark_for_tracking(LAST_CHUNK_IN_FRAME_TOKEN);
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to mark the last chunk for tracking" << std::endl;
            return status;
        }
    }
    status = m_stream->commit_chunk(*m_media_chunk_handler, commit_timestamp_ns);
    if (unlikely(status != ReturnStatus::success)) {
        if (status == ReturnStatus::hw_send_queue_full) {
            m_chunk_pending = true;
            return ReturnStatus::success;
        }
        if (status != ReturnStatus::signal_received) {
            std::cerr << "Failed to commit a chunk " << std::endl;
        }
        return status;
    }
    if (m_chunk_in_field_counter == 0) {
        m_committed_first_chunks++;
    }
    m_chunk_in_field_counter++;
    if (m_chunk_in_field_counter == m_media_settings.chunks_in_frame_field) {
        m_chunk_in_field_counter = 0;
        m_committed_fields++;
    }
    return status;
}

void IPMXStreamSender::print_report_stats()
{
    uint64_t avg_delay = 0;
    if (m_period_sent_frames_cnt) {
        avg_delay = m_period_report_delay_sum / m_period_sent_frames_cnt;
    }
    std::cout << "Stream " << m_stream_number << " SR delay (ns) avg " << avg_delay <<
                 " min " << m_period_report_delay_min <<
                 " max " << m_period_report_delay_max << std::endl;
    m_period_sent_frames_cnt = 0;
    m_period_report_delay_sum = 0;
}

void IPMXStreamSender::init_media_chunk_handler()
{
    m_media_chunk_handler = std::make_unique<MediaChunk>(
            m_stream->get_id(), m_packets_in_chunk, 0); // HDS is not supported
}

void IPMXStreamSender::set_report_chunk_handler(const std::shared_ptr<SharedMessageHandler>& report_handler)
{
    m_report_chunk_handler = report_handler;
}

ReturnStatus IPMXStreamSender::start_media_stream()
{
    return m_stream->create_stream();
}

ReturnStatus IPMXStreamSender::stop_media_stream()
{
        ReturnStatus status = m_media_chunk_handler->cancel_unsent();
        if (status != ReturnStatus::success) {
            return status;
        }
        return m_stream->destroy_stream();
}

IPMXSenderIONode::IPMXSenderIONode(
        const TwoTupleFlow& src_address,
        const std::vector<TwoTupleFlow>& dst_addresses,
        std::shared_ptr<AppSettings>& app_settings,
        size_t index, int cpu_core_affinity,
        time_handler_ns_cb_t nic_time_hanlder_cb,
        time_handler_ns_cb_t wall_time_hanlder_cb) :
    m_media_settings(app_settings->media),
    m_index(index),
    m_sleep_between_operations(app_settings->sleep_between_operations),
    m_print_parameters(app_settings->print_parameters),
    m_cpu_core_affinity(cpu_core_affinity),
    m_packet_payload_size(app_settings->packet_payload_size),
    m_chunks_in_mem_block(app_settings->num_of_chunks_in_mem_block),
    m_packets_in_chunk(app_settings->num_of_packets_in_chunk),
    m_data_stride_size(align_up_pow2(m_packet_payload_size, get_cache_line_size())),
    m_sender_report_buffer_size(align_up_pow2(sizeof(RTCPCompoundPacket), get_cache_line_size())),
    m_get_nic_time_ns(std::move(nic_time_hanlder_cb)),
    m_get_wall_time_ns(std::move(wall_time_hanlder_cb)),
    m_start_send_time_ns(0)
{
    memset(&m_report_mem_region, 0, sizeof(m_report_mem_region));
    initialize_streams(src_address, dst_addresses);
}

std::ostream& IPMXSenderIONode::print(std::ostream& out) const
{
    out << "+#############################################\n"
        << "| Sender index: " << m_index << "\n"
        << "| Thread ID: 0x" << std::hex << std::this_thread::get_id() << std::dec << "\n"
        << "| CPU core affinity: " << m_cpu_core_affinity << "\n"
        << "| Number of streams in this thread: " << m_stream_senders.size() << "\n"
        << "+#############################################\n";
    return out;
}

void IPMXSenderIONode::initialize_streams(
        const TwoTupleFlow& src_address,
        const std::vector<TwoTupleFlow>& dst_addresses)
{
    FourTupleFlow report_route {0, src_address, dst_addresses[0]};
    GenericStreamSettings settings(report_route, false, PacketPacingRate{0, 0},
            dst_addresses.size(), 1, static_cast<uint16_t>(m_sender_report_buffer_size), 0);
    m_rtcp_stream = std::make_shared<GenericSendStream>(settings);

    m_stream_senders.reserve(dst_addresses.size());
    size_t sender_id = 0;
    for (auto & dst_address : dst_addresses) {
        m_stream_senders.emplace_back(sender_id++, src_address, dst_address, m_media_settings,
                m_get_wall_time_ns, m_chunks_in_mem_block, m_packets_in_chunk,
                m_packet_payload_size, m_data_stride_size);
    }
}

ReturnStatus IPMXSenderIONode::query_memory_size(size_t& tx_size) const
{
    tx_size = m_rtcp_stream->get_payload_memory_length();
    return ReturnStatus::success;
};

size_t IPMXSenderIONode::initialize_memory(void* pointer, rmx_mkey_id mkey)
{
    size_t tx_buffer_len = m_rtcp_stream->get_memory_length();
    rmx_mem_region mreg {pointer, tx_buffer_len, mkey};
    m_report_mem_region = mreg;

    return tx_buffer_len;
}

void IPMXSenderIONode::print_parameters()
{
    if (!m_print_parameters) {
        return;
    }

    std::stringstream sender_parameters;
    sender_parameters << this;
    for (auto& stream_sender : m_stream_senders) {
        sender_parameters << stream_sender.get_media_stream();
    }
    std::cout << sender_parameters.str() << std::endl;
}

ReturnStatus IPMXSenderIONode::send_initial_reports()
{
    ReturnStatus status = ReturnStatus::success;
    for (auto& stream_sender : m_stream_senders) {
        stream_sender.set_initial_timestamps(m_start_send_time_ns, get_nic_time_now_ns());
        status = stream_sender.commit_sender_report();
        if (status != ReturnStatus::success) {
            if (status != ReturnStatus::signal_received) {
                std::cerr << "Failed to send an IPMX Sender Report" << std::endl;
            }
            break;
        }
    }
    return status;
}

ReturnStatus IPMXSenderIONode::check_report_completion()
{
    ReturnStatus status = ReturnStatus::success;
    size_t sender_id;
    uint64_t tx_hw_timestamp;

    status = m_report_handler->check_completion(sender_id, tx_hw_timestamp);
    if (status == ReturnStatus::no_completion) {
        return ReturnStatus::success;
    }
    if (status != ReturnStatus::success) {
        return status;
    }

    if (sender_id >= m_stream_senders.size()) {
        std::cerr << "Invalid Send Report completion id" << std::endl;
        return ReturnStatus::failure;
    }

    auto& stream_sender = m_stream_senders[sender_id];
    return stream_sender.notify_report_completion(tx_hw_timestamp);
}

void IPMXSenderIONode::operator()()
{
    set_cpu_resources();
    ReturnStatus status = start_rivermax_streams();
    if (status == ReturnStatus::failure) {
        std::cerr << "Failed to start sender (" << m_index << ") streams" << std::endl;
        return;
    }

    print_parameters();
    prepare_buffers();

    /*
    * Currently the logic in the sender is that all the streams start
    * in the same time and keep aligned during the run. It can be updated if needed.
    */

    uint64_t earliest_start_time_ns = get_nic_time_now_ns();
    for (auto& stream_sender : m_stream_senders) {
        m_start_send_time_ns = stream_sender.calculate_send_time_ns(earliest_start_time_ns);
    }
    size_t committed_frame_field_counter = 0;
    status = ReturnStatus::success;
    uint64_t last_stats_update_time = m_start_send_time_ns;

    wait_until(static_cast<uint64_t>(m_start_send_time_ns -
                                     m_media_settings.frame_field_time_interval_ns));
    send_initial_reports();

    while (likely(status == ReturnStatus::success && SignalHandler::get_received_signal() < 0)) {

        status = check_report_completion();
        if (status != ReturnStatus::success) {
            break;
        }

        uint64_t time_now = get_nic_time_now_ns();
        uint64_t latest_wakeup_time = time_now + m_media_settings.frame_field_time_interval_ns;
        bool all_senders_can_sleep = m_sleep_between_operations;

        for (auto& stream_sender : m_stream_senders) {
            status = stream_sender.track_media_completions();
            if (status != ReturnStatus::success) {
                break;
            }

            if (stream_sender.has_free_frame_buffer()) {
                status = stream_sender.commit_next_media_chunk();

                if (unlikely(status != ReturnStatus::success)) {
                    break;
                }
            }

            if (all_senders_can_sleep) {
                uint64_t wakeup_time;
                if (!stream_sender.can_sleep(wakeup_time)) {
                    all_senders_can_sleep = false;
                } else {
                    latest_wakeup_time = (latest_wakeup_time > wakeup_time) ?
                                         wakeup_time : latest_wakeup_time;
                }
            }
        }

        if (status != ReturnStatus::success) {
            if (status != ReturnStatus::signal_received) {
                std::cerr << "Sending test data failed, aborting" << std::endl;
            }
            break;
        }

        if (all_senders_can_sleep) {
            wait_until(latest_wakeup_time);
        }

        if (time_now > last_stats_update_time + NS_IN_SEC) {
            for (auto& stream_sender : m_stream_senders) {
                stream_sender.print_report_stats();
            }
            last_stats_update_time = time_now;
        }
    }

    status = stop_rivermax_streams();
    if (status != ReturnStatus::success) {
        std::cerr << "Failed to stop sender (" << m_index << ") streams" << std::endl;
        return;
    }
}

ReturnStatus IPMXSenderIONode::start_rivermax_streams()
{
    ReturnStatus status;

    status = m_rtcp_stream->create_stream();
    if (status != ReturnStatus::success) {
        std::cerr << "Failed to create Rivermax generic send stream for RTCP reports" << std::endl;
        return status;
    }
    m_rtcp_stream->initialize_chunks(m_report_mem_region);
    m_report_handler = std::make_shared<SharedMessageHandler>(m_rtcp_stream->get_chunk());

    for (auto& stream_sender : m_stream_senders) {
        status = stream_sender.start_media_stream();
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to create Rivermax media send stream " <<
                         stream_sender.get_stream_number() << std::endl;
            return status;
        }
        stream_sender.init_media_chunk_handler();
        stream_sender.set_report_chunk_handler(m_report_handler);
    }

    return status;
}

ReturnStatus IPMXSenderIONode::stop_rivermax_streams()
{
    ReturnStatus status;

    status = m_rtcp_stream->destroy_stream();
    if (status != ReturnStatus::success) {
            std::cerr << "Failed to destroy Rivermax generic send stream for RTCP reports" << std::endl;
            return status;
        }

    for (auto& stream_sender : m_stream_senders) {
        status = stream_sender.stop_media_stream();
        if (status != ReturnStatus::success) {
            std::cerr << "Failed to destroy Rivermax media send stream " <<
                         stream_sender.get_stream_number() << std::endl;
            return status;
        }
    }

    return ReturnStatus::success;
}

void IPMXSenderIONode::set_cpu_resources()
{
    set_current_thread_affinity(m_cpu_core_affinity);
    rt_set_thread_priority(RMAX_THREAD_PRIORITY_TIME_CRITICAL - 1);
}

inline void IPMXSenderIONode::prepare_buffers()
{
    // TODO: Fill buffers with data, for now, send random garbage as payload.
}

void IPMXSenderIONode::wait_until(uint64_t return_time_ns)
{
    uint64_t time_now_ns = get_nic_time_now_ns();

    if (return_time_ns <= time_now_ns) {
        return;
    }

    size_t sleep_time_ns = return_time_ns - time_now_ns;

    if (sleep_time_ns <= SLEEP_THRESHOLD_NS) {
        return;
    }

    sleep_time_ns -= SLEEP_THRESHOLD_NS;
#ifdef __linux__
    if (m_sleep_between_operations) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(sleep_time_ns));
    } else {
        while (get_nic_time_now_ns() < return_time_ns);
    }
#else
    while (get_nic_time_now_ns() < return_time_ns);
#endif
}
