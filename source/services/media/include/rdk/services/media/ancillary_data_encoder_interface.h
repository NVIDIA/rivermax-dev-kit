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

#ifndef RDK_SERVICES_MEDIA_ANCILLARY_DATA_ENCODER_INTERFACE_H_
#define RDK_SERVICES_MEDIA_ANCILLARY_DATA_ENCODER_INTERFACE_H_

#include <cstddef>
#include <cstdint>

#include "rdk/services/media/ancillary_metadata.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Interface for ancillary data encoders.
 *
 * All ancillary data types (AFD, Timecode, Closed Captions, etc.) implement this interface
 * to provide a unified way of generating ancillary data packets.
 */
class IAncillaryDataEncoder
{
public:
    virtual ~IAncillaryDataEncoder() = default;
    /**
     * @brief: Returns the number of ancillary packets this encoder produces per frame.
     *
     * @return: Number of packets.
     */
    virtual size_t get_packet_count() const { return 1; }
    /**
     * @brief: Writes ancillary data payload to buffer.
     *
     * @param [in] packet_index: Index of packet to write.
     * @param [in] buffer: Destination buffer for the payload data.
     *
     * @return: Size of written data in bytes.
     */
    virtual size_t write_data(size_t packet_index, uint8_t* buffer) = 0;
    /**
     * @brief: Fills descriptor header for this packet.
     *
     * @param [in] packet_index: Index of packet.
     * @param [out] header: Descriptor header to fill.
     */
    virtual void fill_descriptor_header(size_t packet_index, AncillaryDataHeader& header) const = 0;
    /**
     * @brief: Resets encoder state.
     */
    virtual void reset() = 0;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_ANCILLARY_DATA_ENCODER_INTERFACE_H_ */
