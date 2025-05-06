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

#include <string>

#include "CLI/CLI.hpp"

#include "rdk/services/cli/validators.h"
#include "rdk/services/media/media_defs.h"

using namespace rivermax::dev_kit::services;

StreamToThreadsValidator::StreamToThreadsValidator(const size_t& num_of_threads) :
    CLI::Validator("Streams to threads validator")
{
    func_ = [&num_of_threads](std::string& num_of_streams_str) {
        uint16_t num_of_streams;
        bool converted = CLI::detail::lexical_cast(num_of_streams_str, num_of_streams);
        if (!converted) {
            return "Failed parsing num_of_streams " + num_of_streams_str;
        }
        if (num_of_streams < num_of_threads) {
            std::stringstream out;
            out << "Number of streams [" << num_of_streams <<
                "] should be equal or higher number of threads [" << num_of_threads << "]";
            return out.str();
        }
        return std::string();
    };
}

StreamToThreadsFlowsValidator::StreamToThreadsFlowsValidator(
    const size_t& num_of_threads, const size_t& num_of_flows) :
    CLI::Validator("Streams to threads & Flows validator")
{
    func_ = [&num_of_threads, &num_of_flows](std::string& num_of_streams_str) {
        uint16_t num_of_streams;
        bool converted = CLI::detail::lexical_cast(num_of_streams_str, num_of_streams);
        if (!converted) {
            return "Failed parsing num_of_streams " + num_of_streams_str;
        }
        if (num_of_streams < num_of_threads) {
            std::stringstream out;
            out << "Number of streams [" << num_of_streams <<
                "] should be equal or higher number of threads [" << num_of_threads << "]";
            return out.str();
        }
        if (num_of_streams > num_of_flows) {
            std::stringstream out;
            out << "Number of flows [" << num_of_flows <<
                "] should be equal or higher number of streams [" << num_of_streams << "]";
            return out.str();
        }
        return std::string();
    };
}

FlowNumberValidator::FlowNumberValidator(const uint16_t& num_of_streams) :
    CLI::Validator("Validate flow number")
{
    func_ = [&num_of_streams](std::string& num_of_flows_str) {
        int num_of_flows;
        bool converted = CLI::detail::lexical_cast(num_of_flows_str, num_of_flows);
        if (!converted) {
            return "Failed parsing num_of_flows " + num_of_flows_str;
        }
        if (num_of_flows < num_of_streams) {
            std::stringstream out;
            out << "Number of flows [" << num_of_flows <<
                "] should be equal or higher number of streams [" << num_of_streams << "]";
            return out.str();
        }
        return std::string();
    };
}

VideoResolutionValidator::VideoResolutionValidator() :
    CLI::Validator("Video resolution validator")
{
    func_ = [](std::string& resolution_str) -> std::string {
        try {
            Resolution resolution(resolution_str);
            if (resolution.width == 0 || resolution.height == 0) {
                return "Video resolution width and height must be positive integers.";
            }
        } catch (const std::exception&) {
            return "Invalid video resolution format. Expected <width>x<height>";
        }
        return std::string();
    };
}

VideoFrameRateValidator::VideoFrameRateValidator() :
    CLI::Validator("Video frame rate validator")
{
    func_ = [](std::string& frame_rate_str) -> std::string {
        try {
            FrameRate frame_rate(frame_rate_str);
            if (frame_rate.num == 0 || frame_rate.denom == 0) {
                return "Video frame rate numerator and denominator must be positive integers.";
            }
        } catch (const std::exception&) {
            return "Invalid video frame rate format. Expected <numerator>/<denominator> or <integer>";
        }
        return std::string();
    };
}
