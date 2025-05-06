/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef RMAX_APPS_LIB_LIB_SERVICES_CLI_VALIDATORS_H_

#include "CLI/CLI.hpp"

namespace ral
{
namespace lib
{
namespace services
{
/**
 * @brief: Streams to threads validator.
 *
 * The class inherits and implements CLI11 library validator interface.
 * It validates the number of application streams relative to application threads.
 *
 * @param [in] num_of_threads: Number of threads.
 */
class StreamToThreadsValidator : public CLI::Validator
{
public:
    StreamToThreadsValidator(const size_t& num_of_threads);
};
/**
 * @brief: Streams to flows and threads validator.
 *
 * The class inherits and implements CLI11 library validator interface.
 * It validates the number of application streams relative to application flows and threads.
 *
 * @param [in] num_of_threads: Number of threads.
 * @param [in] num_of_flows: Number of flows.
 */
class StreamToThreadsFlowsValidator : public CLI::Validator
{
public:
    StreamToThreadsFlowsValidator(const size_t& num_of_threads, const size_t& num_of_flows);
};
/**
 * @brief: Flows to streams validator.
 *
 * The class inherits and implements CLI11 library validator interface.
 * It validates the number of application flows relative to application streams.
 *
 * @param [in] num_of_threads: Number of threads.
 */
class FlowNumberValidator : public CLI::Validator
{
public:
    FlowNumberValidator(const uint16_t& num_of_streams);
};

} // namespace services
} // namespace lib
} // namespace ral


#define RMAX_APPS_LIB_LIB_SERVICES_CLI_VALIDATORS_H_
#endif /* RMAX_APPS_LIB_LIB_SERVICES_CLI_VALIDATORS_H_ */
