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

#ifndef RDK_APPS_IPMX_RECEIVER_IPMX_RECEIVER_H_
#define RDK_APPS_IPMX_RECEIVER_IPMX_RECEIVER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "rdk/apps/receiver_base_app.h"
#include "rdk/apps/ipmx_receiver/stream_timeline_tracker.h"
#include "rdk/io_node/common/io_node_memory_utils.h"
#include "rdk/io_node/receivers/rtp_receiver_io_node.h"
#include "rdk/services/error_handling/return_status.h"
#include "rdk/core/data_handler/receive_data_consumer_interface.h"

namespace rdk
{
namespace apps
{

/**
 * @brief: This class consumes a chunk of RTCP packets and distributes them to the IPMX
 * Stream Timeline Trackers according to the Receive Flow a packet belongs to.
 */
class RTCPChunkConsumer : public IReceiveDataConsumer
{
private:
    std::vector<std::shared_ptr<IPMXStreamTimelineTracker>> m_trackers;
    const std::shared_ptr<rdk::services::MemoryUtils> m_payload_memory_utils;
    const bool m_rtcp_payload_on_gpu;
    std::vector<std::uint8_t> m_host_chunk_staging;
public:
    /**
     * @brief: RTCPChunkConsumer constructor.
     *
     * @param [in] trackers: An array of pointers to IPMX Stream Timeline Trackers that receive
     * the packets from the chunk, while Flow tag of the packets is used as an index to find the
     * corresponding Tracker.
     * @param [in] payload_memory_utils: Memory utilities to use for handling packet payloads.
     */
    RTCPChunkConsumer(std::vector<std::shared_ptr<IPMXStreamTimelineTracker>>& trackers,
        std::shared_ptr<rdk::services::MemoryUtils> payload_memory_utils) :
        m_trackers(trackers),
        m_payload_memory_utils(std::move(payload_memory_utils)),
        m_rtcp_payload_on_gpu(m_payload_memory_utils->get_memory_location() == rdk::services::MemoryLocation::GPU) {}
    virtual ~RTCPChunkConsumer() = default;
    ReturnStatus consume_chunk(
        const ReceiveChunk& chunk, const IReceiveStream& stream, size_t& consumed_packets) override;
};

/**
 * @brief: This class dispatches an RTP frame event to to the assigened IPMX Stream Timeline Tracker.
 */
class IPMXFrameStartNotifier : public IRTPEventHandler
{
private:
    std::shared_ptr<IPMXStreamTimelineTracker> m_tracker;
public:
    /**
     * @brief: IPMXFrameNotifier constructor.
     *
     * @param [in] tracker: A pointer to IPMX Stream Timeline Tracker that processes the frame events.
     */
    IPMXFrameStartNotifier(std::shared_ptr<IPMXStreamTimelineTracker>& tracker) : m_tracker(tracker) {}
    virtual ~IPMXFrameStartNotifier() = default;
    void notify_rtp_event(uint32_t rtp_timestamp, uint64_t receiver_timestamp) override
    {
        m_tracker->notify_data_frame(rtp_timestamp, receiver_timestamp);
    }
};

/**
 * @brief: Configuration settings for Rivermax Dev Kit IPMX Receiver.
 */
struct IPMXReceiverSettings : AppSettings
{
public:
    static constexpr uint32_t NUM_OF_RTP_PACKETS_IN_RX_BUFFER = 262144;
    static constexpr uint32_t NUM_OF_RTCP_PACKETS_IN_RX_BUFFER_PER_STREAM = 2;
    void init_default_values() override;

    bool is_extended_sequence_number;
    int rtcp_thread_core;
};

/**
 * @brief: Validator for Rivermax Dev Kit IPMX Receiver settings.
 */
class IPMXReceiverSettingsValidator : public ISettingsValidator<IPMXReceiverSettings>
{
public:
    ReturnStatus validate(const IPMXReceiverSettings& settings) const override;
};

/**
 * @brief: CLI settings Builder for Rivermax Dev Kit IPMX Receiver.
 */
class IPMXReceiverCLISettingsBuilder : public CLISettingsBuilder<IPMXReceiverSettings>
{
public:
    /**
     * @brief: IPMXReceiverCLISettingsBuilder constructor.
     *
     * @param [in] argc: Number of CLI arguments.
     * @param [in] argv: CLI arguments strings array.
     * @param [in] app_description: Application description string for the CLI usage.
     * @param [in] app_examples: Application examples string for the CLI usage.
     */
    IPMXReceiverCLISettingsBuilder(int argc, const char** argv,
        const std::string& app_description,
        const std::string& app_examples,
        const ISettingsValidator<IPMXReceiverSettings>& validator) :
        CLISettingsBuilder<IPMXReceiverSettings>(argc, argv, app_description, app_examples, validator) {}
    virtual ~IPMXReceiverCLISettingsBuilder() = default;
protected:
    ReturnStatus add_cli_options(IPMXReceiverSettings& settings) override;
};

using IPMXReceiverUserProvidedSettingsBuilder = UserProvidedSettingsBuilder<IPMXReceiverSettings>;
/**
 * @brief: IPMX receiver application.
 *
 * This is an example of application that uses Rivermax Dev Kit to receive multiple
 * IPMX streams. The number of dropped packets are calculated
 * using RTP sequence numbers. Depending on a command line parameters, the
 * application supports standard RTP sequence numbers or extended sequence
 * numbers as defined in SMPTE ST 2110-20.
 */
class IPMXReceiverApp : public ReceiverBaseApp
{
private:
    /* Settings builder pointer */
    std::unique_ptr<ISettingsBuilder<IPMXReceiverSettings>> m_settings_builder;
    /* Application settings pointer */
    std::shared_ptr<IPMXReceiverSettings> m_ipmx_receiver_settings;
    /* RTCP receiver settings */
    AppSettings m_rtcp_receiver_settings;
    /* Network recv flows */
    std::vector<ReceiveFlow> m_data_flows;
    std::vector<ReceiveFlow> m_rtcp_flows;
    std::vector<std::shared_ptr<IPMXStreamTimelineTracker>> m_ipmx_trackers;
    static constexpr auto STATS_POLLING_PERIOD = std::chrono::nanoseconds{ std::chrono::milliseconds{ 100 } };
public:
    /**
     * @brief: IPMXReceiverApp class constructor.
     *
     * @param [in] settings_builder: Settings builder pointer.
     */
    IPMXReceiverApp(std::unique_ptr<ISettingsBuilder<IPMXReceiverSettings>> settings_builder);
    /**
     * @brief: IPMXReceiverApp class destructor.
     */
    virtual ~IPMXReceiverApp() = default;
private:
    ReturnStatus set_rivermax_clock() final;
    ReturnStatus initialize_app_settings() final;
    /**
     * @brief: Initializes settings of RTCP receiver IO node based on the App settings.
     */
    void initialize_rtcp_ionode_settings();
    void configure_network_flows() final;
    void initialize_receive_io_nodes() final;
    void run_receiver_threads() final;
    void initialize_rtp_streams(RTPReceiverIONode& node, size_t start_id, const std::vector<ReceiveFlow>& flows);
    void initialize_rtcp_stream(RTPReceiverIONode& node, const std::vector<ReceiveFlow>& flows);
};

} // namespace apps
} // namespace rdk

#endif /* RDK_APPS_IPMX_RECEIVER_IPMX_RECEIVER_H_ */
