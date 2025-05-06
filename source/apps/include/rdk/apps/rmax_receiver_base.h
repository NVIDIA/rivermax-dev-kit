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

#ifndef RDK_APPS_RMAX_RECEIVER_BASE_H_
#define RDK_APPS_RMAX_RECEIVER_BASE_H_

#include <cstddef>
#include <vector>
#include <memory>

#include <rivermax_api.h>

#include "rdk/apps/rmax_base_app.h"

using namespace rivermax::dev_kit::io_node;
using namespace rivermax::dev_kit::core;

namespace rivermax
{
namespace dev_kit
{
namespace apps
{
/**
 * @brief: Receiver application base class.
 *
 * This is a base class containing common functions to use in receiver applications.
 */
class RmaxReceiverBaseApp : public RmaxBaseApp
{
protected:
    /* Receiver objects container */
    std::vector<std::shared_ptr<ReceiverIONodeBase>> m_receivers;
    /* NIC device interfaces */
    std::vector<rmx_device_iface> m_device_interfaces;
    /* Is using extended sequence number */
    bool m_is_extended_sequence_number = false;
    /* Number of paths per stream */
    size_t m_num_paths_per_stream = 0;
public:
    virtual ~RmaxReceiverBaseApp() = default;
    ReturnStatus run() override;
    ReturnStatus initialize() override;
    /**
     * @brief: Sets the receive data consumer for the specified stream index.
     *
     * @param [in] stream_index: Stream index.
     * @param [in] data_consumer: Receive data consumer pointer.
     *
     * @return: Status of the operation.
     */
    ReturnStatus set_receive_data_consumer(size_t stream_index, std::unique_ptr<IReceiveDataConsumer> data_consumer);
    /**
     * @brief: Returns application streams total statistics.
     *
     * @tparam StatisticsType: Type of the statistics.
     * @tparam ReceiverStreamType: Type of the receiver stream.
     *
     * @return: Vector of stream statistics.
     */
    template<typename StatisticsType, typename ReceiverStreamType>
    std::vector<StatisticsType> get_streams_total_statistics() const;
protected:
    ReturnStatus initialize_memory_strategy() override;
    /**
     * @brief: RmaxReceiverBaseApp class constructor.
     */
    RmaxReceiverBaseApp();
    /**
     * @brief: Initializes network receive flows.
     *
     * This method initializes the receive flows that will be used
     * in the application. These flows will be distributed
     * in @ref RmaxReceiverBaseApp::distribute_work_for_threads
     * between application threads.
     * The application supports unicast and multicast UDPv4 receive flows.
     */
    virtual void configure_network_flows() = 0;
    /**
     * @brief: Initializes receiver I/O nodes.
     *
     * This method is responsible for initialization of receiver IO node
     * objects to work. It will initiate objects with the relevant parameters.
     * The objects initialized in this method, will be the contexts to the
     * std::thread objects will run in @ref RmaxBaseApp::run_threads
     * method.
     */
    virtual void initialize_receive_io_nodes() = 0;
    /**
     * @brief: Runs application threads.
     */
    virtual void run_receiver_threads() = 0;

private:
    /**
     * @brief: Distributes work for threads.
     *
     * This method is responsible for distributing work to threads, by
     * distributing number of streams per receiver thread uniformly.
     * In future development, this can be extended to different
     * streams per thread distribution policies.
     */
    void distribute_work_for_threads();
};

template<typename StatisticsType, typename ReceiverStreamType>
std::vector<StatisticsType> RmaxReceiverBaseApp::get_streams_total_statistics() const
{
    std::vector<StatisticsType> streams_stats;

    for (const auto& receiver : m_receivers) {
        auto stream_stats = receiver->get_streams_total_statistics<StatisticsType, ReceiverStreamType>();
        streams_stats.insert(streams_stats.end(), stream_stats.begin(), stream_stats.end());
    }

    return streams_stats;
}

} // namespace apps
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_APPS_RMAX_RECEIVER_BASE_H_ */
