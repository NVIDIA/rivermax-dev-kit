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

#include "rdk/services/media/afd_encoder.h"
#include "rdk/services/media/media_defs.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

AFDEncoder::AFDEncoder(AFDCode code, AFDAspectRatio aspect_ratio) :
    m_afd_code(code),
    m_aspect_ratio(aspect_ratio)
{
}

size_t AFDEncoder::write_data(size_t packet_index, uint8_t* buffer)
{
    NOT_IN_USE(packet_index);

    uint8_t afd_byte = (static_cast<uint8_t>(m_afd_code) << AFD_CODE_BIT_SHIFT);
    if (m_aspect_ratio == AFDAspectRatio::Aspect_16x9) {
        afd_byte |= ASPECT_RATIO_16_9_BIT;
    }

    buffer[0] = afd_byte;
    return 1;
}

void AFDEncoder::fill_descriptor_header(size_t packet_index, AncillaryDataHeader& header) const
{
    NOT_IN_USE(packet_index);

    header.did = ANCILLARY_AFD_IDENTIFIER.did;
    header.sdid = ANCILLARY_AFD_IDENTIFIER.sdid;
    header.line_number = LINE_NUMBER;
    header.horizontal_offset = HORIZONTAL_OFFSET;
}

void AFDEncoder::reset()
{
}

} // namespace services
} // namespace dev_kit
} // namespace rivermax
