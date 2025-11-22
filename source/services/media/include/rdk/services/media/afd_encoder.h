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

#ifndef RDK_SERVICES_MEDIA_AFD_ENCODER_H_
#define RDK_SERVICES_MEDIA_AFD_ENCODER_H_

#include <cstdint>

#include "rdk/services/media/ancillary_data_encoder_interface.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: AFD aspect ratio.
 *
 * As defined in SMPTE ST 2016-1, section 5.2.
 */
enum class AFDAspectRatio : uint8_t
{
    Aspect_4x3 = 0,
    Aspect_16x9 = 1
};

/**
 * @brief: Active Format Description (AFD) codes.
 *
 * Subset of all possible AFD codes as defined in SMPTE ST 2016-1, table 1.
 */
enum class AFDCode : uint8_t
{
    FullFrame = 8,
    Pillarbox4x3 = 9,
    FullFrame16x9Protected = 10,
    ShootProtect4x3 = 15
};

/**
 * @brief: Active Format Description (AFD) encoder.
 *
 * Encodes Active Format Description data according to SMPTE 2016-1.
 * AFD describes the aspect ratio and active picture characteristics.
 * Supported codes are a subset of all possible AFD codes.
 * Bar data is not supported.
 *
 * It is based on the SMPTE ST 2016-1 specification.
 */
class AFDEncoder : public IAncillaryDataEncoder
{
public:
    /**
     * @brief: Construct AFD encoder.
     *
     * @param [in] code: AFD code.
     * @param [in] aspect_ratio: Aspect ratio.
     */
    AFDEncoder(AFDCode code, AFDAspectRatio aspect_ratio);
    size_t write_data(size_t packet_index, uint8_t* buffer) override;
    void fill_descriptor_header(size_t packet_index, AncillaryDataHeader& header) const override;
    void reset() override;
    /**
     * @brief: Sets the AFD code.
     *
     * @param [in] code: AFD code.
     */
    void set_afd_code(AFDCode code) { m_afd_code = code; }
    /**
     * @brief: Sets the aspect ratio.
     *
     * @param [in] aspect_ratio: Aspect ratio.
     */
    void set_aspect_ratio(AFDAspectRatio aspect_ratio) { m_aspect_ratio = aspect_ratio; }

private:
    AFDCode m_afd_code;
    AFDAspectRatio m_aspect_ratio;

    static constexpr uint8_t LINE_NUMBER = 9;
    static constexpr uint8_t HORIZONTAL_OFFSET = 0;
    static constexpr uint8_t ASPECT_RATIO_16_9_BIT = 0x04;
    static constexpr uint8_t AFD_CODE_BIT_SHIFT = 3;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_AFD_ENCODER_H_ */
