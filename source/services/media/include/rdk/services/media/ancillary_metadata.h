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

#ifndef RDK_SERVICES_MEDIA_ANCILLARY_METADATA_H_
#define RDK_SERVICES_MEDIA_ANCILLARY_METADATA_H_

#include <cstdint>
#include <vector>

#include "rdk/services/media/media_essence_source.h"

namespace rdk
{
namespace services
{

/**
 * @brief: Describes data header for ancillary data payload.
 */
struct AncillaryDataHeader
{
    bool c_flag = false;                /**< Color channel flag */
    uint16_t line_number = 0;           /**< Line number of the ANC data */
    uint16_t horizontal_offset = 0;     /**< 10-bit words horizontal offset of the ANC data */
    bool s_flag = false;                /**< Data stream flag */
    uint8_t stream_number = 0;          /**< Data stream number */
    uint8_t did = 0;                    /**< Data identification word */
    uint8_t sdid = 0;                   /**< Secondary data identification word */
    uint16_t user_data_words_count = 0; /**< User data 10-bit words count */
};

/**
 * @brief: Contains ancillary data header and user data offset.
 *
 * The offset is used to determine the start of user data in the MediaUnit::data buffer.
 */
struct AncillaryDataDescriptor
{
    AncillaryDataHeader ancillary_data_header;
    size_t user_data_offset = 0;
};

/**
 * @brief: Metadata for ancillary media units.
 *
 * Provides information about the ancillary data packets in the media unit.
 */
struct AncillaryMediaUnitMetadata : public MediaUnitMetadata
{
    /* Ancillary data descriptors. Each entry is a single ancillary data packet. */
    std::vector<AncillaryDataDescriptor> ancillary_data;
    /* Current offset in bytes into the data buffer */
    size_t current_data_offset = 0;

    AncillaryMediaUnitMetadata() :
        MediaUnitMetadata(SMPTEStandard::ST_2110_40)
    {
    }
};

} // namespace services
} // namespace rdk

#endif /* RDK_SERVICES_MEDIA_ANCILLARY_METADATA_H_ */
