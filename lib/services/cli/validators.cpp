/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <string>

#include "CLI/CLI.hpp"

#include "services/cli/validators.h"

using namespace ral::lib::services;

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
