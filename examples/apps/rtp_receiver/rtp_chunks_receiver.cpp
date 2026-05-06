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

#include <memory>

#include "rdk/apps/rtp_receiver/rtp_receiver.h"
#include "rdk/examples/apps/rtp_receiver/rtp_chunks_receiver.h"

using namespace rdk::apps;
using namespace rdk::services;
using namespace rdk::examples::apps;

namespace rdk
{
namespace examples
{
namespace apps
{

constexpr const char* EXAMPLE_DESCRIPTION = "NVIDIA Rivermax Dev Kit RTP Chunks Receiver Example";
constexpr const char* EXAMPLE_USAGE = "Examples:\n"
                                      "  1. rtp_chunks_receiver --local-ip 1.2.3.4 --source-ip 5.6.7.8";

RTPChunksReceiverExample::RTPChunksReceiverExample() :
    BaseExample(EXAMPLE_DESCRIPTION, EXAMPLE_USAGE)
{
}

void RTPChunksReceiverExample::add_cli_options()
{
    BaseExample::add_cli_options();

    m_cli_parser_manager->add_option(CLIOptStr::SRC_IP);
}

void RTPChunksReceiverExample::configure_rtp_receiver_settings(RTPReceiverSettings& settings) const
{

    /** Configure network settings */
    settings.local_ip = m_settings->local_ip;
    settings.source_ip = m_settings->source_ip;
    settings.destination_ip = "224.1.1.1";
    settings.destination_port = 2000;

    /** Configure receive settings */
    settings.is_extended_sequence_number = false;
    settings.num_of_packets_in_chunk = RTPReceiverSettings::DEFAULT_NUM_OF_PACKETS_IN_CHUNK;

    /** Configure thread/stream settings */
    settings.num_of_threads = 1;
    settings.num_of_total_streams = 1;
    settings.internal_thread_core = 0;
    settings.app_threads_cores = {1};

    /** Configure memory settings */
    settings.allocator_type = AllocatorTypeUI::Auto;
    settings.app_memory_alloc = true;
    settings.header_data_split = true;
}

ReturnStatus RTPChunksReceiverExample::operator()()
{
    /** 1. Configure RTP receiver settings with user parameters */
    RTPReceiverSettings rtp_receiver_settings;

    rtp_receiver_settings.init_default_values();
    configure_rtp_receiver_settings(rtp_receiver_settings);

    /** 2. Set up RTP receiver settings validation */
    RTPReceiverSettingsValidator settings_validator;

    /** 3. Configure user-provided settings builder with validator */
    auto settings_builder = std::make_unique<RTPReceiverUserProvidedSettingsBuilder>(
        std::move(rtp_receiver_settings), settings_validator);

    /** 4. Initialize the RTP Receiver application */
    RTPReceiverApp app(std::move(settings_builder));
    auto status = app.initialize();
    RETURN_FAILURE_ON_ERROR(status, "Failed to initialize RTP Receiver application");

    /**
     * 4. Initialize a custom data consumer.
     *
     * Create a dummy data consumer using @ref NullReceiveDataConsumer simulating the reception of RTP packets
     * without actual processing. This consumer implements the @ref IReceiveDataConsumer interface,
     * which client application should implement to handle received data.
     */
    auto dummy_data_consumer = std::make_unique<NullReceiveDataConsumer>();

    /** 5. Set the custom data consumer to the RTP Receiver application */
    constexpr size_t stream_index = 0;
    status = app.set_receive_data_consumer(stream_index, std::move(dummy_data_consumer));
    RETURN_FAILURE_ON_ERROR(status, "Failed to set data consumer");

    /** 6. Start receiving RTP chunks */
    status = app.run();
    RETURN_FAILURE_ON_ERROR(status, "Failed to run RTP Receiver application");

    return ReturnStatus::success;
}

} // namespace apps
} // namespace examples
} // namespace rdk

int main(int argc, const char* argv[])
{
    return rdk::examples::common_example_main<RTPChunksReceiverExample>(argc, argv);
}
