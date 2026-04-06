/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef RDK_SERVICES_MEDIA_CLOSED_CAPTION_MOCK_SOURCE_H_
#define RDK_SERVICES_MEDIA_CLOSED_CAPTION_MOCK_SOURCE_H_

#include "rdk/services/media/closed_caption_source_interface.h"
#include "rdk/services/media/media_defs.h"

namespace rdk
{
namespace services
{

/**
 * @brief: Mock closed caption source for testing.
 *
 * Generates a simple test pattern with:
 * - 2 seconds ON: Shows "CAPTION XX" (incrementing counter)
 * - 2 seconds OFF: No caption
 */
class ClosedCaptionMockSource : public IClosedCaptionSource
{
public:
    std::string get_caption_text(uint64_t timestamp_ns) override;

private:
    /** Cycle duration constants. */
    /* Full cycle duration (ON + OFF). */
    static constexpr uint64_t CYCLE_DURATION_NS = 4 * NS_IN_SEC;
    /* Caption ON duration. */
    static constexpr uint64_t ON_DURATION_NS = 2 * NS_IN_SEC;
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_MEDIA_CLOSED_CAPTION_MOCK_SOURCE_H_ */
