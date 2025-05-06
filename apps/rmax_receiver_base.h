/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef RMAX_APPS_LIB_APPS_RMAX_RECEIVER_BASE_H_
#define RMAX_APPS_LIB_APPS_RMAX_RECEIVER_BASE_H_

#include "rmax_base_app.h"

using namespace ral::lib::core;

namespace ral
{
namespace apps
{

/**
 * @brief: Receiver application base class.
 *
 * This is a base class containing common functions to use in receiver
 * applications.
 */
class RmaxReceiverBaseApp : public RmaxBaseApp
{
protected:
    /* Stream per thread distribution */
    std::unordered_map<size_t, size_t> m_streams_per_thread;
    /* Size of header memory buffer */
    size_t m_header_mem_size = 0;
    /* Size of payload memory buffer */
    size_t m_payload_mem_size = 0;
    /* Buffer for packet header (if header-data split is enabled) */
    byte_t* m_header_buffer = nullptr;
    /* Buffer for packet payload */
    byte_t* m_payload_buffer = nullptr;

public:
    /**
     * @brief: RmaxReceiverBaseApp class constructor.
     *
     * @param [in] app_description: Application description string for the CLI usage.
     * @param [in] app_examples: Application examples string for the CLI usage.
     */
    RmaxReceiverBaseApp(const std::string& app_description, const std::string& app_examples);
    virtual ~RmaxReceiverBaseApp() = default;
    ReturnStatus run() override;
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
    /**
     * @brief: Allocates application memory.
     *
     * This method is responsible for allocation of the required memory for
     * the application using @ref ral::lib::services::MemoryAllocator interface.
     * The allocation policy of the application is allocating one big memory
     * block. This memory block will be distributed to the different
     * components of the application.
     *
     * @return: Returns status of the operation.
     */
    ReturnStatus allocate_app_memory();
    /**
     * @brief: Allocates memory and aligns it to page size.
     *
     * @param [in]  header_size: Requested header memory size.
     * @param [in]  payload_size: Requested payload memory size.
     * @param [out] header: Allocated header memory pointer.
     * @param [out] payload: Allocated payload memory pointer.
     *
     * @return: True if successful.
     */
    bool allocate_and_align(size_t header_size, size_t payload_size, byte_t*& header, byte_t*& payload);
protected:
    /**
     * @brief: Initializes network receive flows.
     *
     * This method initializes the receive flows that will be used
     * in the application. These flows will be distributed
     * in @ref ral::apps::RmaxReceiverBaseApp::distribute_work_for_threads
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
     * std::thread objects will run in @ref ral::apps::RmaxBaseApp::run_threads
     * method.
     */
    virtual void initialize_receive_io_nodes() = 0;
    /**
     * @brief: Registers previously allocated memory if requested.
     *
     * If @ref m_register_memory is set then this function registers
     * application memory using @ref rmax_register_memory.
     *
     * @return: Returns status of the operation.
     */
    virtual ReturnStatus register_app_memory() = 0;
    /**
     * @brief: Unregister previously registered memory.
     *
     * Unregister memory using @ref rmax_deregister_memory.
     */
    virtual void unregister_app_memory() = 0;
    /**
     * @brief: Distributes memory for receivers.
     *
     * This method is responsible for distributing the memory allocated
     * by @ref allocate_app_memory to the receivers of the application.
     */
    virtual void distribute_memory_for_receivers() = 0;
    /**
     * @brief: Returns the memory size for all the receive streams.
     *
     * This method calculates the sum of memory sizes for all IONodes and their streams.
     *
     * @param [out] hdr_mem_size: Required header memory size.
     * @param [out] pld_mem_size: Required payload memory size.
     *
     * @return: Return status of the operation.
     */
    virtual ReturnStatus get_total_streams_memory_size(size_t& hdr_mem_size, size_t& pld_mem_size) = 0;
    /**
     * @brief: Cleans up Rivermax library resources.
     *
     * @retun: Status of the operation.
     */
    ReturnStatus cleanup_rivermax_resources() override;
    /**
     * @brief: Runs application threads.
     */
    virtual void run_receiver_threads() = 0;
};

} // namespace apps
} // namespace ral

#endif /* RMAX_APPS_LIB_APPS_RMAX_RECEIVER_BASE_H_ */
