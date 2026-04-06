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

#ifndef RDK_EXAMPLES_SERVICES_SDP_SDP_SMPTE_2110_30_DESCRIPTION_H_
#define RDK_EXAMPLES_SERVICES_SDP_SDP_SMPTE_2110_30_DESCRIPTION_H_

#include "rdk/services/error_handling/return_status.h"
#include "rdk/examples/base_example.h"

using namespace rdk::services;

namespace rdk
{
namespace examples
{
namespace services
{

/**
 * @brief: SMPTE 2110-30 SDP generation example.
 *
 * This example demonstrates the generation of SMPTE 2110-30 SDP string by a client application.
 * It showcases a small subset of the supported SDP attributes settings.
 * The full list of supported attributes can be found in @ref SMPTE2110_30_MediaDescription.
 */
class SMPTE_2110_30_SDPGenerationExample : public BaseExample
{
public:
    /**
     * @brief: Constructor for the @ref SMPTE_2110_30_SDPGenerationExample class.
     */
    SMPTE_2110_30_SDPGenerationExample();

    ReturnStatus operator()() override;
    void add_cli_options() override {};
};

} // namespace services
} // namespace examples
} // namespace rdk

#endif /* RDK_EXAMPLES_SERVICES_SDP_SDP_SMPTE_2110_30_DESCRIPTION_H_ */
