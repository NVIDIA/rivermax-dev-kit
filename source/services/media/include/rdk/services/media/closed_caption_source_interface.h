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

#ifndef RDK_SERVICES_MEDIA_CLOSED_CAPTION_SOURCE_INTERFACE_H_
#define RDK_SERVICES_MEDIA_CLOSED_CAPTION_SOURCE_INTERFACE_H_

#include <cstdint>
#include <string>

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Interface for closed caption text sources.
 *
 * Provides caption text based on timestamp.
 */
class IClosedCaptionSource
{
public:
    virtual ~IClosedCaptionSource() = default;
    /**
     * @brief: Get caption text for the specified timestamp.
     *
     * @param [in] timestamp_ns: Timestamp in nanoseconds.
     *
     * @return: Caption text to display.
     */
    virtual std::string get_caption_text(uint64_t timestamp_ns) = 0;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_CLOSED_CAPTION_SOURCE_INTERFACE_H_ */
