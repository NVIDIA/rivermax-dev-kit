/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <sstream>

#include "rt_threads.h"

#include "rdk/apps/media_receiver/media_receiver.h"
#include "rdk/io_node/receivers/rtp_receiver_io_node.h"
#include "rdk/services/cli/options.h"
#include "rdk/services/media/media_settings_ancillary.h"
#include "rdk/services/media/media_settings_audio.h"
#include "rdk/services/media/media_settings_video.h"
#include "rdk/services/media/null_essence_sink.h"
#include "rdk/core/flow/receive_flow.h"

using namespace rdk::apps;
using rdk::services::IP_OCTET_LEN;


MediaReconstructionDataConsumer::MediaReconstructionDataConsumer(
    // TODO: std::unique_ptr<IULPPacketBufferReader> buffer_reader,
    std::shared_ptr<IMediaEssenceSink> sink,
    std::shared_ptr<MediaUnitPool> pool)
    // TODO: : m_buffer_reader(std::move(buffer_reader)),
      : m_sink(std::move(sink)),
      m_media_unit_pool(std::move(pool)),
      m_current_media_unit(nullptr)
{
}

ReturnStatus MediaReconstructionDataConsumer::consume_chunk(
    const ReceiveChunk& chunk, const IReceiveStream& stream, size_t& consumed_packets)
{
    // TODO: Implement reconstruction
    NOT_IN_USE(stream);
    consumed_packets = chunk.get_length();
    return ReturnStatus::success;
}

void MediaReceiverSettings::init_default_values()
{
    AppSettings::init_default_values();
    app_memory_alloc = true;
    media.frames_fields_in_mem_block = MediaSettings::DEFAULT_NUM_OF_MEDIA_UNITS_IN_MEM_BLOCK;
}

ReturnStatus MediaReceiverSettingsValidator::validate(const MediaReceiverSettings& settings) const
{
    ReturnStatus rc = ValidatorUtils::validate_ip4_address(settings.source_ip);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_ip4_address(settings.local_ip);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_ip4_address(settings.destination_ip);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_ip4_port(settings.destination_port);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_core(settings.internal_thread_core);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_core(settings.app_threads_cores);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_gpu_direct_header_size_compatibility(settings.gpu_id, settings.packet_app_header_size);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    if (!settings.media.enable_video && !settings.media.enable_audio && !settings.media.enable_ancillary) {
        std::cerr << "At least one media type must be enabled: "
                  << CLIOptStr::ENABLE_VIDEO << ", "
                  << CLIOptStr::ENABLE_AUDIO << ", or "
                  << CLIOptStr::ENABLE_ANCILLARY << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

ReturnStatus MediaReceiverCLISettingsBuilder::add_cli_options(MediaReceiverSettings& settings)
{
    if (m_cli_parser_manager == nullptr) {
        std::cerr << "CLI parser manager is not initialized" << std::endl;
        return ReturnStatus::failure;
    }
    m_cli_parser_manager->add_option(CLIOptStr::SRC_IP);
    m_cli_parser_manager->add_option(CLIOptStr::DST_IP);
    m_cli_parser_manager->add_option(CLIOptStr::LOCAL_IP);
    m_cli_parser_manager->add_option(CLIOptStr::DST_PORT);
    m_cli_parser_manager->add_option(CLIOptStr::THREADS);
    m_cli_parser_manager->add_option(CLIOptStr::STREAMS);
    m_cli_parser_manager->add_option(CLIOptStr::PAYLOAD_SIZE);
    m_cli_parser_manager->add_option(CLIOptStr::APP_HDR_SIZE);
    m_cli_parser_manager->add_option(CLIOptStr::INTERNAL_CORE);
    m_cli_parser_manager->add_option(CLIOptStr::APPLICATION_CORE);
#ifdef CUDA_ENABLED
    m_cli_parser_manager->add_option(CLIOptStr::GPU_ID);
    m_cli_parser_manager->add_option(CLIOptStr::LOCK_GPU_CLOCKS);
#endif
    m_cli_parser_manager->add_option(CLIOptStr::ALLOCATOR_TYPE);
    m_cli_parser_manager->add_option(CLIOptStr::REGISTER_MEMORY);
    m_cli_parser_manager->add_option(CLIOptStr::VERBOSE);
    m_cli_parser_manager->add_option(CLIOptStr::STATS_REPORT_INTERVAL);
    m_cli_parser_manager->add_option(CLIOptStr::ENABLE_VIDEO)->group(CLIGroupStr::VIDEO_FORMAT_OPTIONS);
    m_cli_parser_manager->add_option(CLIOptStr::VIDEO_RESOLUTION)->group(CLIGroupStr::VIDEO_FORMAT_OPTIONS);
    m_cli_parser_manager->add_option(CLIOptStr::VIDEO_FRAME_RATE)->group(CLIGroupStr::VIDEO_FORMAT_OPTIONS);
    m_cli_parser_manager->add_option(CLIOptStr::VIDEO_SAMPLING)->group(CLIGroupStr::VIDEO_FORMAT_OPTIONS);
    m_cli_parser_manager->add_option(CLIOptStr::VIDEO_SCAN_TYPE)->group(CLIGroupStr::VIDEO_FORMAT_OPTIONS);
    m_cli_parser_manager->add_option(CLIOptStr::VIDEO_BIT_DEPTH)->group(CLIGroupStr::VIDEO_FORMAT_OPTIONS);
    m_cli_parser_manager->add_option(CLIOptStr::ENABLE_AUDIO)->group(CLIGroupStr::AUDIO_FORMAT_OPTIONS);
    m_cli_parser_manager->add_option(CLIOptStr::PTIME_US)->group(CLIGroupStr::AUDIO_FORMAT_OPTIONS);
    m_cli_parser_manager->add_option(CLIOptStr::AUDIO_SAMPLING_RATE)->group(CLIGroupStr::AUDIO_FORMAT_OPTIONS);
    m_cli_parser_manager->add_option(CLIOptStr::AUDIO_ENCODING)->group(CLIGroupStr::AUDIO_FORMAT_OPTIONS);
    m_cli_parser_manager->add_option(CLIOptStr::ENABLE_ANCILLARY)->group(CLIGroupStr::ANCILLARY_FORMAT_OPTIONS);
    m_cli_parser_manager->add_option(CLIOptStr::ANCILLARY_TYPES)->group(CLIGroupStr::ANCILLARY_FORMAT_OPTIONS);
    m_cli_parser_manager->add_option(CLIOptStr::ANCILLARY_DATA_WORDS_COUNT)->group(CLIGroupStr::ANCILLARY_FORMAT_OPTIONS);
    return ReturnStatus::success;
}

const std::unordered_map<SMPTEStandard, std::function<ReturnStatus(MediaReceiverApp*)>>
    MediaReceiverApp::s_smpte_standard_config_map = {
        {SMPTEStandard::ST_2110_20, [](MediaReceiverApp* app) { return app->configure_video_types(); }},
        {SMPTEStandard::ST_2110_30, [](MediaReceiverApp* app) { return app->configure_audio_types(); }},
        {SMPTEStandard::ST_2110_40, [](MediaReceiverApp* app) { return app->configure_ancillary_types(); }},
};

MediaReceiverApp::MediaReceiverApp(std::unique_ptr<ISettingsBuilder<MediaReceiverSettings>> settings_builder)
    : ReceiverBaseApp(),
      m_settings_builder(std::move(settings_builder))
{
    m_num_paths_per_stream = 1;
}

ReturnStatus MediaReceiverApp::initialize_app_settings()
{
    if (m_settings_builder == nullptr) {
        std::cerr << "Settings builder is not initialized" << std::endl;
        return ReturnStatus::failure;
    }
    m_media_receiver_settings = std::make_shared<MediaReceiverSettings>();
    ReturnStatus rc = m_settings_builder->build(*m_media_receiver_settings);
    if (rc == ReturnStatus::success) {
        m_app_settings = m_media_receiver_settings;
        return ReturnStatus::success;
    }
    if (rc != ReturnStatus::success_cli_help) {
        std::cerr << "Failed to build settings" << std::endl;
    }
    m_obj_init_status = rc;
    return rc;
}

void MediaReceiverApp::run_receiver_threads()
{
    run_threads(m_receivers);
}

ReturnStatus MediaReceiverApp::post_load_settings()
{
    if (m_media_receiver_settings->media.enable_video) {
        m_media_receiver_settings->enabled_smpte_standards.insert(SMPTEStandard::ST_2110_20);
    }
    if (m_media_receiver_settings->media.enable_audio) {
        m_media_receiver_settings->enabled_smpte_standards.insert(SMPTEStandard::ST_2110_30);
    }
    if (m_media_receiver_settings->media.enable_ancillary) {
        m_media_receiver_settings->enabled_smpte_standards.insert(SMPTEStandard::ST_2110_40);
    }
    ReturnStatus rc = configure_smpte_standards_processing();
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to configure SMPTE standards processing" << std::endl;
        return rc;
    }
    size_t total_streams = 0;
    for (const auto& node : m_media_receiver_settings->smpte_standard_to_nodes) {
        total_streams += node.second;
    }
    m_app_settings->num_of_total_streams = static_cast<uint16_t>(total_streams);

    return ReturnStatus::success;
}

ReturnStatus MediaReceiverApp::initialize_smpte_standards()
{
    return configure_smpte_standards_processing();
}

void MediaReceiverApp::configure_network_flows()
{
    std::string ip_prefix_str;
    int ip_last_octet;
    uint16_t src_port = 0;

    auto ip_vec = CLI::detail::split(m_app_settings->destination_ip, '.');
    ip_prefix_str = std::string(ip_vec[0] + "." + ip_vec[1] + "." + ip_vec[2] + ".");
    ip_last_octet = std::stoi(ip_vec[3]);

    m_flows.reserve(m_app_settings->num_of_total_streams);

    size_t id = 0;
    for (size_t flow_index = 0; flow_index < m_app_settings->num_of_total_streams; ++flow_index) {
        std::ostringstream ip;
        ip << ip_prefix_str << (ip_last_octet + flow_index) % IP_OCTET_LEN;
        ReceiveFlow flow(id++, m_app_settings->source_ip, src_port, ip.str(), m_app_settings->destination_port);
        m_flows.push_back(flow);
    }
}

void MediaReceiverApp::initialize_receive_io_nodes()
{
    size_t streams_offset = 0;
    for (size_t rx_idx = 0; rx_idx < m_app_settings->num_of_threads; rx_idx++) {
        const auto& node = m_media_receiver_settings->smpte_standard_to_nodes[rx_idx];
        const MediaSettings& media_settings = node.first;
        SMPTEStandard config_smpte_standard = media_settings.get_smpte_standard();
        bool is_extended_seq = (config_smpte_standard == SMPTEStandard::ST_2110_20 ||
                               config_smpte_standard == SMPTEStandard::ST_2110_40);

        int recv_cpu_core = m_app_settings->app_threads_cores[rx_idx % m_app_settings->app_threads_cores.size()];

        auto flows = std::vector<ReceiveFlow>(
            m_flows.begin() + streams_offset,
            m_flows.begin() + streams_offset + m_streams_per_thread[rx_idx]);
        m_receivers.push_back(std::unique_ptr<ReceiverIONodeBase>(new RTPReceiverIONode(
            *m_app_settings,
            is_extended_seq,
            { m_app_settings->local_ip },
            rx_idx,
            recv_cpu_core,
            *m_memory_utils,
            false)));
        static_cast<RTPReceiverIONode*>(m_receivers[rx_idx].get())->initialize_streams(streams_offset, flows);
        m_receivers[rx_idx]->set_statistics_report_interval(m_app_settings->stats_report_interval_ms);
        streams_offset += m_streams_per_thread[rx_idx];
    }
}

void MediaReceiverApp::distribute_work_for_threads()
{
    const size_t num_receivers = m_media_receiver_settings->smpte_standard_to_nodes.size();
    m_app_settings->num_of_threads = num_receivers;
    m_streams_per_thread.clear();
    for (size_t rx_idx = 0; rx_idx < num_receivers; rx_idx++) {
        m_streams_per_thread[rx_idx] = m_media_receiver_settings->smpte_standard_to_nodes[rx_idx].second;
    }
}

template <typename SettingsType>
ReturnStatus MediaReceiverApp::configure_media_type_helper(std::unique_ptr<SettingsType> settings,
    const std::string& smpte_standard_name)
{
    size_t num_of_threads = std::min<size_t>(m_app_settings->num_of_threads,
        m_app_settings->num_of_total_streams);
    size_t min_number_streams_per_thread =
        m_app_settings->num_of_total_streams / num_of_threads;

    ReturnStatus rc = settings->create_default_calculator();
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to create default calculator for " << smpte_standard_name << std::endl;
        return rc;
    }

    for (size_t idx = 0; idx < num_of_threads; idx++) {
        size_t num_of_streams_in_cur_thread;
        if (min_number_streams_per_thread * num_of_threads + idx <
            m_app_settings->num_of_total_streams) {
            num_of_streams_in_cur_thread = min_number_streams_per_thread + 1;
        } else {
            num_of_streams_in_cur_thread = min_number_streams_per_thread;
        }
        m_media_receiver_settings->smpte_standard_to_nodes.emplace_back(
            *settings, num_of_streams_in_cur_thread);
    }
    m_media_receiver_settings->smpte_standard_configs.push_back(std::move(settings));
    return ReturnStatus::success;
}

ReturnStatus MediaReceiverApp::configure_video_types()
{
    if (!m_app_settings->media.enable_video) {
        std::cerr << "Video is not enabled" << std::endl;
        return ReturnStatus::failure;
    }
    auto video_settings = std::make_unique<SMPTE_2110_20_MediaSettings>(*m_app_settings);
    return configure_media_type_helper(std::move(video_settings), "2110-20");
}

ReturnStatus MediaReceiverApp::configure_audio_types()
{
    if (!m_app_settings->media.enable_audio) {
        std::cerr << "Audio is not enabled" << std::endl;
        return ReturnStatus::failure;
    }
    auto audio_settings = std::make_unique<SMPTE_2110_30_MediaSettings>(*m_app_settings);
    return configure_media_type_helper(std::move(audio_settings), "2110-30");
}

ReturnStatus MediaReceiverApp::configure_ancillary_types()
{
    if (!m_app_settings->media.enable_ancillary) {
        std::cerr << "Ancillary is not enabled" << std::endl;
        return ReturnStatus::failure;
    }
    auto ancillary_settings = std::make_unique<SMPTE_2110_40_MediaSettings>(*m_app_settings);
    return configure_media_type_helper(std::move(ancillary_settings), "2110-40");
}

ReturnStatus MediaReceiverApp::configure_smpte_standards_processing()
{
    if (m_media_receiver_settings->enabled_smpte_standards.empty()) {
        std::cerr << "No SMPTE standards are enabled" << std::endl;
        return ReturnStatus::failure;
    }
    m_media_receiver_settings->smpte_standard_to_nodes.clear();
    m_media_receiver_settings->smpte_standard_configs.clear();
    for (const auto& smpte_standard : m_media_receiver_settings->enabled_smpte_standards) {
        auto it = s_smpte_standard_config_map.find(smpte_standard);
        if (it == s_smpte_standard_config_map.end()) {
            std::cerr << "Unsupported SMPTE standard: " << static_cast<int>(smpte_standard) << std::endl;
            return ReturnStatus::failure;
        }
        ReturnStatus rc = it->second(this);
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to configure SMPTE standard: " << static_cast<int>(smpte_standard) << std::endl;
            return rc;
        }
    }
    return ReturnStatus::success;
}

ReturnStatus MediaReceiverApp::initialize()
{
    ReturnStatus rc = ReceiverBaseApp::initialize();
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to initialize receiver base app" << std::endl;
        return rc;
    }

    rc = set_internal_media_essence_sinks();
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to set internal media essence sinks" << std::endl;
        return rc;
    }

    return ReturnStatus::success;
}

std::unique_ptr<MediaReconstructionDataConsumer> MediaReceiverApp::create_consumer_for_stream(
    const MediaSettings& media_settings,
    std::shared_ptr<IMediaEssenceSink> sink)
{
    SMPTEStandard smpte_standard = media_settings.get_smpte_standard();
    size_t media_unit_count = media_settings.media_units_in_mem_block;
    size_t media_unit_size = media_settings.bytes_per_media_unit;

    auto pool = std::make_shared<MediaUnitPool>(media_unit_count, media_unit_size, smpte_standard,
        *m_payload_allocator);
    // TODO: Uncomment when RTP media packet buffer reader is implemented.
    // auto buffer_reader = create_rtp_media_packet_buffer_reader(
    //     smpte_standard, media_settings,
    //     m_header_allocator->get_memory_utils(),
    //     m_payload_allocator->get_memory_utils());
    // if (!buffer_reader) {
    //     std::cerr << "Failed to create packet buffer reader" << std::endl;
    //     return nullptr;
    // }
    return std::make_unique<MediaReconstructionDataConsumer>(
        // TODO: std::move(buffer_reader),
        std::move(sink), pool);
}

ReturnStatus MediaReceiverApp::set_internal_media_essence_sinks()
{
    for (size_t rx_idx = 0; rx_idx < m_app_settings->num_of_threads; rx_idx++) {
        const auto& node = m_media_receiver_settings->smpte_standard_to_nodes[rx_idx];
        const MediaSettings& media_settings = node.first;

        for (size_t stream_idx = 0; stream_idx < m_streams_per_thread[rx_idx]; stream_idx++) {
            auto default_sink = std::make_shared<NullEssenceSink>();
            auto consumer = create_consumer_for_stream(media_settings, default_sink);
            if (!consumer) {
                std::cerr << "Failed to create consumer for receiver " << rx_idx
                          << " stream " << stream_idx << std::endl;
                return ReturnStatus::failure;
            }
            ReturnStatus rc = m_receivers[rx_idx]->set_receive_data_consumer(stream_idx, std::move(consumer));
            if (rc != ReturnStatus::success) {
                std::cerr << "Failed to set data consumer for receiver " << rx_idx
                          << " stream " << stream_idx << std::endl;
                return rc;
            }
        }
    }
    return ReturnStatus::success;
}

ReturnStatus MediaReceiverApp::set_media_essence_sink(
    size_t stream_index,
    SMPTEStandard smpte_standard,
    std::shared_ptr<IMediaEssenceSink> essence_sink)
{
    if (!essence_sink) {
        return ReturnStatus::success;
    }
    size_t thread_index = 0;
    size_t internal_stream_index = 0;
    auto rc = find_internal_stream_index(stream_index, thread_index, internal_stream_index);
    if (rc != ReturnStatus::success) {
        std::cerr << "Error setting media essence sink, invalid stream index " << stream_index << std::endl;
        return ReturnStatus::failure;
    }
    const MediaSettings& media_settings = m_media_receiver_settings->smpte_standard_to_nodes[thread_index].first;
    SMPTEStandard stream_smpte_standard = media_settings.get_smpte_standard();
    if (stream_smpte_standard != smpte_standard) {
        std::cerr << "Stream " << stream_index
                  << " has SMPTE standard " << static_cast<int>(stream_smpte_standard)
                  << ", expected " << static_cast<int>(smpte_standard) << std::endl;
        return ReturnStatus::failure;
    }
    auto consumer = create_consumer_for_stream(media_settings, std::move(essence_sink));
    if (!consumer) {
        std::cerr << "Failed to create consumer for stream " << stream_index << std::endl;
        return ReturnStatus::failure;
    }
    rc = m_receivers[thread_index]->set_receive_data_consumer(internal_stream_index, std::move(consumer));
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to set media essence sink for stream " << stream_index << std::endl;
    }
    return rc;
}
