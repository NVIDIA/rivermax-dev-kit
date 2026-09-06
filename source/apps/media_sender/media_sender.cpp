/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <functional>
#include <string>
#include <unordered_map>

#include "rdk/services/cli/options.h"

#include "rdk/facade.h"
#include "rdk/apps/media_sender/media_sender.h"
#include "rdk/apps/base_memory_strategy.h"
#include "rdk/services/media/ancillary_essence_source.h"
#include "rdk/services/media/media_settings_ancillary.h"
#include "rdk/services/media/media_settings_audio.h"
#include "rdk/services/media/media_settings_video.h"
#include "rdk/services/protocol/ipv4.h"
#include "rdk/services/sdp/sdp_defs.h"
#include "rdk/services/utils/clock.h"
#include "rdk/services/utils/defs.h"

using namespace rdk::apps;

void MediaSenderSettings::init_default_values()
{
    AppSettings::init_default_values();
    destination_ip = "";
    destination_port = 0;
    destination_ips = {DESTINATION_IP_DEFAULT};
    destination_ports = {DESTINATION_PORT_DEFAULT};
    media.frames_fields_in_mem_block = MediaSenderSettings::DEFAULT_FRAME_FIELDS_IN_MEM_BLOCK;
    media.resolution = { _1080_WIDTH, _1080_HEIGHT };
    num_of_packets_in_chunk = 0;
}

ReturnStatus MediaSenderSettingsValidator::validate(const MediaSenderSettings& settings) const
{
    if (!settings.local_ip.empty() || !settings.destination_ip.empty()) {
        std::cerr << "Use of single parameters is deprecated. Please use plural parameters instead" << std::endl;
        return ReturnStatus::failure;
    }
    if (settings.local_ips.empty()) {
        std::cerr << "At least one local IP must be specified" << std::endl;
        return ReturnStatus::failure;
    }
    if (settings.enable_redundancy && settings.local_ips.size() < 2 ) {
        std::cerr << "SMPTE 2022-7 redundancy requires at least two local IP addresses" << std::endl;
        return ReturnStatus::failure;
    }
    if (settings.local_ips.size() > 1 && !settings.enable_redundancy) {
        std::cerr << "Only one local IP address is supported when SMPTE 2022-7 redundancy is disabled" << std::endl;
        return ReturnStatus::failure;
    }
    if (settings.local_ips.size() > rdk::RivermaxDevKitFacade::get_max_redundant_streams()) {
        std::cerr << "Up to " << rdk::RivermaxDevKitFacade::get_max_redundant_streams()
                  << " local IP addresses are supported" << std::endl;
        return ReturnStatus::failure;
    }
    if (settings.destination_ips.size() != settings.local_ips.size()) {
        std::cerr << "Must be the same number of destination IPs as number of local IPs" << std::endl;
        return ReturnStatus::failure;
    }
    if (settings.destination_ports.size() != settings.local_ips.size()) {
        std::cerr << "Must be the same number of destination ports as number of local IPs" << std::endl;
        return ReturnStatus::failure;
    }
    ReturnStatus rc = ValidatorUtils::validate_ip4_address(settings.local_ips);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_ip4_address(settings.destination_ips);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_ip4_port(settings.destination_ports);
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
    rc = ValidatorUtils::validate_core(settings.statistics_reader_core);
    if (rc != ReturnStatus::success) {
        return rc;
    }
    rc = ValidatorUtils::validate_gpu_direct_header_split_compatibility(settings.gpu_id, settings.header_data_split);
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
    if (settings.media.enable_video && !settings.media.enable_extended_video_formats) {
        auto is_supported = [](const auto& values, const auto& value) {
            return std::find(values.begin(), values.end(), value) != values.end();
        };
        if (!is_supported(SUPPORTED_VIDEO_RESOLUTIONS, settings.media.resolution)) {
            std::cerr << "Error: Video resolution " << settings.media.resolution
                      << " requires extended video formats" << std::endl;
            return ReturnStatus::failure;
        }
        if (!is_supported(SUPPORTED_VIDEO_BIT_DEPTHS, settings.media.color_bit_depth)) {
            std::cerr << "Error: Video bit depth " << enum_to_string(settings.media.color_bit_depth)
                      << " requires extended video formats" << std::endl;
            return ReturnStatus::failure;
        }
        if (settings.media.alpha_bit_depth != VideoBitDepth::Unknown &&
            !is_supported(SUPPORTED_VIDEO_BIT_DEPTHS, settings.media.alpha_bit_depth)) {
            std::cerr << "Error: Alpha bit depth " << enum_to_string(settings.media.alpha_bit_depth)
                      << " requires extended video formats" << std::endl;
            return ReturnStatus::failure;
        }
    }
    bool media_file_used = (!settings.video_file.empty() || !settings.audio_file.empty());
    bool valid_memory_strategy = settings.app_memory_alloc || settings.dynamic_media_file_load;
    if (media_file_used && !valid_memory_strategy) {
        std::cerr << "Error: Application memory allocation or dynamic file loading must"
                     " be enabled when media files are used" << std::endl;
        return ReturnStatus::failure;
    }
    return ReturnStatus::success;
}

ReturnStatus MediaSenderCLISettingsBuilder::add_cli_options(MediaSenderSettings& settings)
{
    if (m_cli_parser_manager == nullptr) {
        std::cerr << "CLI parser manager is not initialized" << std::endl;
        return ReturnStatus::failure;
    }

    ReturnStatus rc = ReturnStatus::success;

    // Adds a CLI option and records a failure if it could not be registered.
    auto add_option = [&](const std::string& option_name) -> CLI::Option* {
        auto* option = m_cli_parser_manager->add_option(option_name);
        if (option == nullptr) {
            rc = ReturnStatus::failure;
        }
        return option;
    };

    auto add_grouped_option = [&](const std::string& option_name, const std::string& group) {
        auto* option = add_option(option_name);
        if (option) {
            option->group(group);
        }
        return option;
    };
    auto add_extended_video_format_option = [&](const std::string& option_name) {
        if (auto* option = add_grouped_option(option_name, CLIGroupStr::VIDEO_FORMAT_OPTIONS)) {
            option->description(option->get_description() + " (use --evf for extended)");
        }
    };

    try {
        if (auto* option = add_option(CLIOptStr::LOCAL_IPS)) {
            option->required();
        }
        add_option(CLIOptStr::DST_IPS);
        add_option(CLIOptStr::DST_PORTS);
        add_option(CLIOptStr::THREADS);
        if (auto* option = add_option(CLIOptStr::STREAMS)) {
            option->check(StreamToThreadsValidator(settings.num_of_threads));
        }
        add_option(CLIOptStr::VERBOSE);
        add_option(CLIOptStr::INTERNAL_CORE);
        add_option(CLIOptStr::APPLICATION_CORE);
        add_option(CLIOptStr::SLEEP);
        auto* hds = add_option(CLIOptStr::HEADER_DATA_SPLIT);
#ifdef CUDA_ENABLED
        if (auto* option = add_option(CLIOptStr::GPU_ID)) {
            if (hds != nullptr) {
                option->needs(hds);
            }
        }
        add_option(CLIOptStr::LOCK_GPU_CLOCKS);
#else
        NOT_IN_USE(hds);
#endif
        auto* stats_enabled = add_option(CLIOptStr::ENABLE_STATS_READER);
        if (auto* option = add_option(CLIOptStr::STATS_CORE)) {
            option->needs(stats_enabled);
        }
        if (auto* option = add_option(CLIOptStr::STATS_SESSION_ID)) {
            option->needs(stats_enabled);
        }
        auto* mem = add_option(CLIOptStr::APP_MEMORY_ALLOC);
        if (auto* option = add_option(CLIOptStr::ALLOCATOR_TYPE)) {
            option->needs(mem);
        }
        if (auto* option = add_option(CLIOptStr::REGISTER_MEMORY)) {
            option->needs(mem);
        }
        add_option(CLIOptStr::ENABLE_REDUNDANCY);
        add_grouped_option(CLIOptStr::ENABLE_VIDEO, CLIGroupStr::VIDEO_FORMAT_OPTIONS);
        add_grouped_option(CLIOptStr::ENABLE_ALPHA, CLIGroupStr::VIDEO_FORMAT_OPTIONS);
        add_grouped_option(CLIOptStr::ENABLE_EXTENDED_VIDEO_FORMATS, CLIGroupStr::VIDEO_FORMAT_OPTIONS);
        add_grouped_option(CLIOptStr::VIDEO_FILE, CLIGroupStr::VIDEO_FORMAT_OPTIONS);
        add_option(CLIOptStr::DYNAMIC_FILE_LOADING);
        add_extended_video_format_option(CLIOptStr::VIDEO_RESOLUTION);
        add_grouped_option(CLIOptStr::VIDEO_FRAME_RATE, CLIGroupStr::VIDEO_FORMAT_OPTIONS);
        add_grouped_option(CLIOptStr::VIDEO_SAMPLING, CLIGroupStr::VIDEO_FORMAT_OPTIONS);
        add_grouped_option(CLIOptStr::VIDEO_SCAN_TYPE, CLIGroupStr::VIDEO_FORMAT_OPTIONS);
        add_extended_video_format_option(CLIOptStr::VIDEO_BIT_DEPTH);
        add_extended_video_format_option(CLIOptStr::ALPHA_BIT_DEPTH);
        add_option(CLIOptStr::PACKETS);
        add_grouped_option(CLIOptStr::ENABLE_AUDIO, CLIGroupStr::AUDIO_FORMAT_OPTIONS);
        add_grouped_option(CLIOptStr::AUDIO_FILE, CLIGroupStr::AUDIO_FORMAT_OPTIONS);
        add_grouped_option(CLIOptStr::PTIME_US, CLIGroupStr::AUDIO_FORMAT_OPTIONS);
        add_grouped_option(CLIOptStr::AUDIO_SAMPLING_RATE, CLIGroupStr::AUDIO_FORMAT_OPTIONS);
        add_grouped_option(CLIOptStr::AUDIO_ENCODING, CLIGroupStr::AUDIO_FORMAT_OPTIONS);
        add_grouped_option(CLIOptStr::ENABLE_ANCILLARY, CLIGroupStr::ANCILLARY_FORMAT_OPTIONS);
        add_grouped_option(CLIOptStr::ANCILLARY_FILE, CLIGroupStr::ANCILLARY_FORMAT_OPTIONS);
        add_grouped_option(CLIOptStr::ANCILLARY_TYPES, CLIGroupStr::ANCILLARY_FORMAT_OPTIONS);
        add_grouped_option(CLIOptStr::ANCILLARY_DATA_WORDS_COUNT, CLIGroupStr::ANCILLARY_FORMAT_OPTIONS);
    } catch (const CLI::Error& error) {
        std::cerr << "Failed to register CLI options: " << error.what() << std::endl;
        return ReturnStatus::failure;
    }

    return rc;
}

MediaSenderApp::MediaSenderApp(std::unique_ptr<ISettingsBuilder<MediaSenderSettings>> settings_builder) :
    BaseApp(),
    m_settings_builder(std::move(settings_builder))
{
}

ReturnStatus MediaSenderApp::post_load_settings()
{
    if(m_app_settings->media.enable_video) {
        m_media_sender_settings->enabled_smpte_standards.insert(SMPTEStandard::ST_2110_20);
    }
    if (m_app_settings->media.enable_alpha) {
        if (m_app_settings->media.alpha_bit_depth == VideoBitDepth::Unknown) {
            m_app_settings->media.alpha_bit_depth = m_app_settings->media.color_bit_depth;
        }
    }
    if(m_app_settings->media.enable_audio) {
        m_media_sender_settings->enabled_smpte_standards.insert(SMPTEStandard::ST_2110_30);
    }
    if(m_app_settings->media.enable_ancillary) {
        m_media_sender_settings->enabled_smpte_standards.insert(SMPTEStandard::ST_2110_40);
    }
    m_app_settings->num_of_total_flows = m_app_settings->num_of_total_streams;
    // A requested chunk size wins, a known resolution brings its own, and the rest
    // is left to the calculator, which derives one from the video geometry.
    if (m_app_settings->num_of_packets_in_chunk) {
        m_app_settings->num_of_packets_in_chunk_specified = true;
    } else {
        auto default_packets_in_chunk = MediaSenderSettings::find_default_packets_in_chunk(
            m_app_settings->media.resolution);
        if (default_packets_in_chunk) {
            m_app_settings->num_of_packets_in_chunk = *default_packets_in_chunk;
            m_app_settings->num_of_packets_in_chunk_specified = true;
        }
    }
    return ReturnStatus::success;
}

ReturnStatus MediaSenderApp::initialize_app_settings()
{
    if (m_settings_builder == nullptr) {
        std::cerr << "Settings builder is not initialized" << std::endl;
        return ReturnStatus::failure;
    }
    m_media_sender_settings = std::make_shared<MediaSenderSettings>();
    ReturnStatus rc = m_settings_builder->build(*m_media_sender_settings);
    if (rc == ReturnStatus::success) {
        m_app_settings = m_media_sender_settings;
        return ReturnStatus::success;
    }
    if (rc != ReturnStatus::success_cli_help) {
        std::cerr << "Failed to build settings" << std::endl;
    }
    m_obj_init_status = rc;
    return rc;
}

ReturnStatus MediaSenderApp::initialize()
{
    ReturnStatus rc  = BaseApp::initialize();
    if (rc != ReturnStatus::success) {
        return rc;
    }

    try {
        distribute_work_for_threads();
        rc = configure_smpte_standards_processing();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to configure SMPTE standards" << std::endl;
            return rc;
        }
        configure_network_flows();
        rc = initialize_sender_threads();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to initialize sender threads" << std::endl;
            return rc;
        }
        rc = set_internal_media_essence_sources();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to set internal media essence sources" << std::endl;
            return rc;
        }
        rc = configure_memory_layout();
        if (rc == ReturnStatus::failure) {
            std::cerr << "Failed to configure memory layout" << std::endl;
            return rc;
        }
    }
    catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return ReturnStatus::failure;
    }

    m_obj_init_status = ReturnStatus::obj_init_success;
    return ReturnStatus::success;
}

ReturnStatus MediaSenderApp::initialize_memory_strategy()
{
    auto base_memory_strategy = std::make_unique<BaseMemoryStrategy>(
        *m_header_allocator, *m_payload_allocator, *m_auxiliary_allocator,
        *m_memory_utils,
        m_device_interfaces,
        m_num_paths_per_stream,
        m_app_settings->app_memory_alloc,
        m_app_settings->register_memory);

    for (const auto& sender : m_senders) {
        base_memory_strategy->add_memory_subcomponent(sender);
    }

    m_memory_strategy = std::move(base_memory_strategy);

    return ReturnStatus::success;
}

std::vector<MediaSenderIONodeSettings> MediaSenderApp::get_io_nodes_settings() const
{
    std::vector<MediaSenderIONodeSettings> settings;

    settings.reserve(m_senders.size());
    for (const auto& sender : m_senders) {
        settings.push_back(sender->get_settings());
    }
    return settings;
}

void MediaSenderApp::stop_io_nodes() noexcept
{
    BaseApp::stop_io_nodes(m_senders);
    if (m_synchronizer) {
        m_synchronizer->stop();
    }
}

size_t MediaSenderApp::num_io_nodes() const
{
    return m_senders.size();
}

ReturnStatus MediaSenderApp::stop_io_node(size_t id)
{
    return BaseApp::stop_io_node(m_senders, id);
}

ReturnStatus MediaSenderApp::await_io_node(size_t id, std::chrono::nanoseconds timeout) const
{
    return BaseApp::await_io_node(m_senders, id, timeout);
}

ReturnStatus MediaSenderApp::do_work()
{
    if (m_obj_init_status != ReturnStatus::obj_init_success) {
        return m_obj_init_status;
    }

    ReturnStatus rc = run_stats_reader();
    if (rc == ReturnStatus::failure) {
        return ReturnStatus::failure;
    }

    try {
        run_threads(m_senders);
    }
    catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

ReturnStatus MediaSenderApp::set_rivermax_clock()
{
    ReturnStatus rc = set_rivermax_ptp_clock(&m_device_interfaces[0]);
    if(rc == ReturnStatus::success) {
        uint64_t ptp_time = 0;
        rc = get_rivermax_ptp_time_ns(ptp_time);
    }
    return rc;
}

void MediaSenderApp::configure_network_flows()
{
    m_num_paths_per_stream = m_app_settings->local_ips.size();
    m_flows.reserve(m_app_settings->num_of_total_flows * m_num_paths_per_stream);

    std::vector<std::string> ip_prefix_str;
    std::vector<uint8_t> ip_last_octet;
    for (const auto& dst_ip : m_app_settings->destination_ips) {
        auto ip_vec = CLI::detail::split(dst_ip, '.');
        ip_prefix_str.push_back(std::string(ip_vec[0] + "." + ip_vec[1] + "." + ip_vec[2] + "."));
        ip_last_octet.push_back(static_cast<uint8_t>(std::stoi(ip_vec[3])));
    }

    size_t flow_index = 0;
    std::ostringstream ip;
    uint16_t port;

    size_t total_num_of_flows = 0;
    for (const auto& node : m_media_sender_settings->smpte_standard_to_nodes) {
        total_num_of_flows += node.second;
    }
    m_flows.reserve(total_num_of_flows);

    for (const auto& node : m_media_sender_settings->smpte_standard_to_nodes) {
        auto& num_of_streams = node.second;
        for (size_t stream_index = 0; stream_index < num_of_streams; stream_index++) {
            for (size_t path_index = 0; path_index < m_num_paths_per_stream; path_index++) {
                ip << ip_prefix_str[path_index] << (ip_last_octet[path_index] + flow_index) % IP_OCTET_LEN;
                port = m_app_settings->destination_ports[path_index];
                m_flows.push_back(FourTupleFlow(flow_index, m_app_settings->local_ips[path_index], m_app_settings->source_port, ip.str(), port));
                ip.str("");
            }
            flow_index++;
        }
    }
}

template<typename SettingsType>
ReturnStatus MediaSenderApp::configure_media_type_helper(
    std::shared_ptr<SettingsType> settings,
    const std::string& smpte_standard_name)
{
    size_t min_number_streams_per_thread =
        m_app_settings->num_of_total_streams / m_app_settings->num_of_threads;

    ReturnStatus rc = settings->create_default_calculator();
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to create default calculator for "
                  << smpte_standard_name << std::endl;
        return rc;
    }

    for (size_t idx = 0; idx < m_app_settings->num_of_threads; idx++) {
        size_t num_of_streams_in_cur_thread;
        if (min_number_streams_per_thread * m_app_settings->num_of_threads + idx <
            m_app_settings->num_of_total_streams) {
            num_of_streams_in_cur_thread = min_number_streams_per_thread + 1;
        } else {
            num_of_streams_in_cur_thread = min_number_streams_per_thread;
        }
        m_media_sender_settings->smpte_standard_to_nodes.emplace_back(
            settings, num_of_streams_in_cur_thread);
    }

    return ReturnStatus::success;
}

ReturnStatus MediaSenderApp::configure_video_types()
{
    if (!m_app_settings->media.enable_video) {
        return ReturnStatus::failure;
    }

    auto warn_about_short_packets = [](const SMPTE_2110_20_MediaSettings& settings, const char* stream_type) {
        /* ST 2110-20 advises avoiding IP datagrams under 1000 bytes. */
        constexpr size_t small_ip_datagram_threshold = 1000;
        const size_t ip_datagram_size = IPV4_MIN_HEADER_SIZE + UDP_HEADER_SIZE +
            settings.protocol_header_size + settings.raw_packet_payload_size;
        if (settings.packets_in_media_unit > 1 && ip_datagram_size < small_ip_datagram_threshold) {
            std::cerr << "Warning: current packetization produces small IP datagrams ("
                      << ip_datagram_size << " bytes) for this " << stream_type
                      << std::endl;
        }
    };

    auto video_settings = std::make_shared<SMPTE_2110_20_MediaSettings>(*m_app_settings);
    ReturnStatus rc = configure_media_type_helper(video_settings, "2110-20");
    if (rc != ReturnStatus::success) {
        return rc;
    }
    warn_about_short_packets(*video_settings, "2110-20 video");

    if (m_app_settings->media.alpha_bit_depth != VideoBitDepth::Unknown) {
        auto alpha_settings = std::make_shared<SMPTE_2110_20_MediaSettings>(
            *m_app_settings, true);
        rc = configure_media_type_helper(alpha_settings, "2110-20 alpha/key");
        if (rc == ReturnStatus::success) {
            warn_about_short_packets(*alpha_settings, "2110-20 alpha/key");
        }
    }

    return rc;
}

ReturnStatus MediaSenderApp::configure_audio_types()
{
    if (!m_app_settings->media.enable_audio) {
        return ReturnStatus::failure;
    }

    auto audio_settings = std::make_shared<SMPTE_2110_30_MediaSettings>(*m_app_settings);
    return configure_media_type_helper(
        std::move(audio_settings), "2110-30 audio");
}

ReturnStatus MediaSenderApp::configure_ancillary_types()
{
    if (!m_app_settings->media.enable_ancillary) {
        return ReturnStatus::failure;
    }

    auto ancillary_settings = std::make_shared<SMPTE_2110_40_MediaSettings>(*m_app_settings);
    return configure_media_type_helper(
        std::move(ancillary_settings), "2110-40 ancillary");
}

const std::unordered_map<SMPTEStandard, std::function<ReturnStatus(MediaSenderApp*)>> MediaSenderApp::s_smpte_standard_config_map = {
    {SMPTEStandard::ST_2110_20, [](MediaSenderApp* app) { return app->configure_video_types(); }},
    {SMPTEStandard::ST_2110_30, [](MediaSenderApp* app) { return app->configure_audio_types(); }},
    {SMPTEStandard::ST_2110_40, [](MediaSenderApp* app) { return app->configure_ancillary_types(); }},
};

ReturnStatus MediaSenderApp::configure_smpte_standards_processing()
{
    if (m_media_sender_settings->enabled_smpte_standards.empty()) {
        std::cerr << "No SMPTE standards are enabled" << std::endl;
        return ReturnStatus::failure;
    }

    m_media_sender_settings->smpte_standard_to_nodes.clear();
    ReturnStatus rc = ReturnStatus::success;
    for (const auto& smpte_standard : m_media_sender_settings->enabled_smpte_standards) {
        auto it = s_smpte_standard_config_map.find(smpte_standard);
        if (it == s_smpte_standard_config_map.end()) {
            std::cerr << "Unsupported SMPTE standard: " << static_cast<int>(smpte_standard) << std::endl;
            return ReturnStatus::failure;
        }
        rc = it->second(this);
        if (rc != ReturnStatus::success) {
            return rc;
        }
    }
    return rc;
}

void MediaSenderApp::distribute_work_for_threads()
{
    size_t num_of_threads = std::min<size_t>(
        m_app_settings->num_of_threads,
        m_app_settings->num_of_total_streams);

    if (num_of_threads < m_app_settings->num_of_threads) {
        std::cout << "The number of media sender threads is limited to the number of streams ("
                  << num_of_threads << ")" << std::endl;
        m_app_settings->num_of_threads = num_of_threads;
    }

    m_streams_per_thread.reserve(m_app_settings->num_of_threads);
    for (int stream = 0; stream < m_app_settings->num_of_total_streams; stream++) {
        m_streams_per_thread[stream % m_app_settings->num_of_threads]++;
    }
}

ReturnStatus MediaSenderApp::initialize_sender_threads()
{
    size_t streams_offset = 0;
    size_t sender_idx = 0;
    m_synchronizer = std::make_shared<LinearSynchronizer>(m_media_sender_settings->smpte_standard_to_nodes.size());
    for (const auto& node : m_media_sender_settings->smpte_standard_to_nodes) {
        auto& smpte_standard_config = node.first;
        auto num_of_streams = node.second;
        int sender_cpu_core;
        if (sender_idx < m_app_settings->app_threads_cores.size()) {
            sender_cpu_core = m_app_settings->app_threads_cores[sender_idx];
        } else {
            std::cerr << "Warning: CPU affinity for Sender " << sender_idx <<
                         " is not set!!!" << std::endl;
            sender_cpu_core = NO_CPU_AFFINITY;
        }
        auto flows = std::vector<FourTupleFlow>(
            m_flows.begin() + streams_offset,
            m_flows.begin() + streams_offset + num_of_streams * m_num_paths_per_stream);
        m_senders.push_back(std::make_unique<MediaSenderIONode>(
            m_num_paths_per_stream,
            m_app_settings,
            smpte_standard_config,
            sender_idx,
            num_of_streams,
            sender_cpu_core,
            *m_memory_utils,
            MediaSenderApp::get_time_ns));
        m_senders[sender_idx]->initialize_send_flows(flows);
        ReturnStatus rc = m_senders[sender_idx]->initialize_streams();
        if (rc != ReturnStatus::success) {
            std::cerr << "Failed to initialize streams for sender " << sender_idx << std::endl;
            return rc;
        }
        m_senders[sender_idx]->set_synchronizer(m_synchronizer);
        streams_offset += num_of_streams * m_num_paths_per_stream;
        sender_idx++;
    }
    return ReturnStatus::success;
}

ReturnStatus MediaSenderApp::set_media_essence_sources(
        size_t stream_index,
        SMPTEStandard smpte_standard,
        std::shared_ptr<IMediaEssenceSource> preload_essence_source,
        std::shared_ptr<IMediaEssenceSource> runtime_essence_source,
        bool runtime_contains_payload)
{
    size_t sender_thread_index = 0;
    size_t sender_stream_index = 0;

    auto rc = find_internal_stream_index(stream_index, sender_thread_index, sender_stream_index);
    if (rc != ReturnStatus::success) {
        std::cerr << "Error setting media essence source, invalid stream index " << stream_index << std::endl;
        return rc;
    }

    rc = m_senders[sender_thread_index]->set_media_essence_sources(
        sender_stream_index, smpte_standard,
        std::move(preload_essence_source), std::move(runtime_essence_source),
        runtime_contains_payload);

    if (rc != ReturnStatus::success) {
        std::cerr << "Error setting media essence source for stream "
                  << sender_stream_index << " on sender " << sender_thread_index << std::endl;
    }

    return rc;
}

ReturnStatus MediaSenderApp::set_internal_media_essence_sources()
{
    std::shared_ptr<IMediaEssenceSource> essence_source;
    ReturnStatus rc;
    bool contains_payload = true;
    size_t sender_index = 0;
    for (const auto& node : m_media_sender_settings->smpte_standard_to_nodes) {
        const auto& smpte_standard_config = *node.first;
        auto num_of_streams = node.second;
        for(size_t stream_index = 0; stream_index < num_of_streams; stream_index++) {
            if (smpte_standard_config.get_smpte_standard() == SMPTEStandard::ST_2110_40) {
                const auto& ancillary_settings = static_cast<const SMPTE_2110_40_MediaSettings&>(smpte_standard_config);
                essence_source = std::make_shared<AncillaryEssenceSource>(ancillary_settings);
                rc = m_senders[sender_index]->set_media_essence_sources(
                    stream_index,
                    smpte_standard_config.get_smpte_standard(),
                    nullptr,
                    std::move(essence_source));
            } else {
                if (smpte_standard_config.media_file.empty()) {
                    essence_source = std::make_shared<NullEssenceSource>(smpte_standard_config);
                } else {
                    auto media_file_essence_source = std::make_shared<MediaFileEssenceSource>(
                        smpte_standard_config.media_file,
                        smpte_standard_config.get_smpte_standard(),
                        smpte_standard_config.bytes_per_media_unit,
                        *m_header_allocator,
                        true);
                    rc = media_file_essence_source->load_media_units();
                    if (rc != ReturnStatus::success) {
                        std::cerr << "Failed to load media units from file: " << smpte_standard_config.media_file
                                  << std::endl;
                        return rc;
                    }
                    essence_source = std::move(media_file_essence_source);
                }
                if (smpte_standard_config.dynamic_media_file_load) {
                    rc = m_senders[sender_index]->set_media_essence_sources(
                        stream_index,
                        smpte_standard_config.get_smpte_standard(),
                        nullptr,
                        std::move(essence_source));
                } else {
                    rc = m_senders[sender_index]->set_media_essence_sources(
                        stream_index,
                        smpte_standard_config.get_smpte_standard(),
                        std::move(essence_source));
                }
            }
            if (rc != ReturnStatus::success) {
                std::cerr << "Error setting media essence source for stream " << stream_index << " on sender "
                          << sender_index << std::endl;
                return rc;
            }
        }
        sender_index++;
    }
    return ReturnStatus::success;
}

uint64_t MediaSenderApp::get_time_ns(void* context)
{
    NOT_IN_USE(context);
    uint64_t ptp_time = 0;
    ReturnStatus rc = get_rivermax_ptp_time_ns(ptp_time);
    if (rc != ReturnStatus::success) {
        std::cerr << "Failed to get PTP time" << std::endl;
        return 0;
    }
    return ptp_time;
}
