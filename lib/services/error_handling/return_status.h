/*
 * Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef RMAX_APPS_LIB_LIB_SERVICES_ERROR_HANDLING_RETURN_STATUS_H_
#define RMAX_APPS_LIB_LIB_SERVICES_ERROR_HANDLING_RETURN_STATUS_H_

namespace ral
{
namespace lib
{
namespace services
{

/**
 * @brief: Return status ENUM class.
 *
 * The return status will be used to pass function call exit status
 * between the different components of the library.
 */
enum class ReturnStatus
{
    success,
    failure,
    success_cli_help,
    signal_received,
    obj_init_success,
    obj_init_failure,
    hw_send_queue_full,
    memory_allocation_failure,
    rmax_version_incompatible,
    rmax_version_unaligned,
    no_free_chunks,
    no_completion,
};

} // namespace services
} // namespace lib
} // namespace ral

#endif /* RMAX_APPS_LIB_LIB_SERVICES_ERROR_HANDLING_RETURN_STATUS_H_ */
