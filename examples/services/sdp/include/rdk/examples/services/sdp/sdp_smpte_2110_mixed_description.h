/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef RDK_EXAMPLES_SERVICES_SDP_SDP_SMPTE_2110_MIXED_DESCRIPTION_H_
#define RDK_EXAMPLES_SERVICES_SDP_SDP_SMPTE_2110_MIXED_DESCRIPTION_H_

#include "rdk/services/error_handling/return_status.h"
#include "rdk/examples/base_example.h"

using namespace rivermax::dev_kit::services;

namespace rivermax
{
namespace dev_kit
{
namespace examples
{
namespace services
{

/**
 * @brief: SMPTE 2110 mixed media SDP generation example.
 *
 * This example demonstrates the generation of SMPTE 2110 SDP string with multiple media descriptions
 * by a client application. It showcases video (2110-20), audio (2110-30), and ancillary (2110-40)
 * streams in a single SDP.
 * The full list of supported attributes can be found in @ref SMPTE2110_20_MediaDescription,
 * @ref SMPTE2110_30_MediaDescription, and @ref SMPTE2110_40_MediaDescription.
 */
class SMPTE_2110_Mixed_SDPGenerationExample : public BaseExample
{
public:
    /**
     * @brief: Constructor for the @ref SMPTE_2110_Mixed_SDPGenerationExample class.
     */
    SMPTE_2110_Mixed_SDPGenerationExample();

    ReturnStatus operator()() override;
    void add_cli_options() override {};
};

} // namespace services
} // namespace examples
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_EXAMPLES_SERVICES_SDP_SDP_SMPTE_2110_MIXED_DESCRIPTION_H_ */
