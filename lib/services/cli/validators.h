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
