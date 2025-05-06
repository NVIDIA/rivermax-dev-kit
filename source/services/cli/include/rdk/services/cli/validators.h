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

#ifndef RDK_SERVICES_CLI_VALIDATORS_H_
#define RDK_SERVICES_CLI_VALIDATORS_H_

#include <cstddef>

#include "CLI/CLI.hpp"

namespace rivermax
{
namespace dev_kit
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
/**
 * @brief: Video resolution validator.
 *
 * The class inherits and implements CLI11 library validator interface.
 * It validates the video resolution format <width>x<height>.
 */
class VideoResolutionValidator : public CLI::Validator
{
public:
    VideoResolutionValidator();
};
/**
 * @brief: Video frame rate validator.
 *
 * The class inherits and implements CLI11 library validator interface.
 * It validates the video frame rate format <numerator>/<denominator> or <integer>.
 */
class VideoFrameRateValidator : public CLI::Validator
{
public:
    VideoFrameRateValidator();
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_CLI_VALIDATORS_H_ */
