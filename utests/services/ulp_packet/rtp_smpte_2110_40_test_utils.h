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

#ifndef RDK_UTESTS_SERVICES_ULP_PACKET_RTP_SMPTE_2110_40_TEST_UTILS_H_
#define RDK_UTESTS_SERVICES_ULP_PACKET_RTP_SMPTE_2110_40_TEST_UTILS_H_

#include <vector>

#include <gtest/gtest.h>

#include "rdk/services/ulp_packet/writers/rtp_smpte_2110_40_packet_writer.h"
#include "rdk/services/ulp_packet/smpte_2110_40_ancillary_utils.h"
#include "rdk/services/ulp_packet/rtp_smpte_2110_40_packet_context.h"
#include "rdk/services/media/ancillary_metadata.h"
#include "rdk/services/media/media_defs.h"
#include "rdk/services/error_handling/return_status.h"

// Intentional in a test-only header: bring service types into scope for all test utilities below.
using namespace rdk::services; // NOLINT(google-build-using-namespace)

/**
 * @brief: Constructs an AncillaryDataDescriptor with the given field values.
 *
 * @param [in] did: Data identifier byte.
 * @param [in] sdid: Secondary data identifier byte.
 * @param [in] user_data_words_count: Number of user data words.
 * @param [in] line_number: Source video line number (default 0).
 * @param [in] horizontal_offset: Horizontal offset in the line (default 0).
 * @param [in] c_flag: Color channel flag (default false).
 * @param [in] s_flag: Stream number valid flag (default false).
 * @param [in] stream_number: Stream number (default 0).
 * @param [in] user_data_offset: Byte offset into the user data words buffer (default 0).
 *
 * @return: Populated AncillaryDataDescriptor.
 */
static AncillaryDataDescriptor make_descriptor(
    uint8_t did, uint8_t sdid, uint16_t user_data_words_count,
    uint16_t line_number = 0,
    uint16_t horizontal_offset = 0,
    bool c_flag = false,
    bool s_flag = false,
    uint8_t stream_number = 0,
    size_t user_data_offset = 0)
{
    AncillaryDataDescriptor descriptor{};
    descriptor.ancillary_data_header.did = did;
    descriptor.ancillary_data_header.sdid = sdid;
    descriptor.ancillary_data_header.user_data_words_count = user_data_words_count;
    descriptor.ancillary_data_header.line_number = line_number;
    descriptor.ancillary_data_header.horizontal_offset = horizontal_offset;
    descriptor.ancillary_data_header.c_flag = c_flag;
    descriptor.ancillary_data_header.s_flag = s_flag;
    descriptor.ancillary_data_header.stream_number = stream_number;
    descriptor.user_data_offset = user_data_offset;
    return descriptor;
}

/**
 * @brief: Builds a complete ST 2110-40 RTP packet into buffer using the writer.
 *
 * Computes the payload size from the descriptor list, allocates buffer, then
 * calls fill_header and fill_payload. Calls ASSERT_EQ internally; must be
 * called from a void test context.
 *
 * @param [out] buffer: Receives the encoded RTP packet bytes.
 * @param [in] descriptors: ANC sub-packet descriptors to encode.
 * @param [in] user_data_words_bytes: User data word bytes for all sub-packets.
 * @param [in] sequence: RTP sequence number (default 1).
 * @param [in] timestamp: RTP timestamp (default 0).
 * @param [in] ssrc: Synchronization source identifier (default 0).
 * @param [in] extended_sequence: 32-bit extended sequence number (default 1).
 * @param [in] field_indicator: Field indicator value (default progressive).
 */
static void build_rtp_packet(
    std::vector<uint8_t>& buffer,
    const std::vector<AncillaryDataDescriptor>& descriptors,
    const std::vector<uint8_t>& user_data_words_bytes,
    uint16_t sequence = 1,
    uint32_t timestamp = 0,
    uint32_t ssrc = 0,
    uint32_t extended_sequence = 1,
    uint8_t field_indicator = RTP_2110_40_FIELD_INDICATOR_PROGRESSIVE)
{
    size_t payload_bytes = 0;
    for (const auto& descriptor : descriptors) {
        payload_bytes +=
            AncillaryDataUtils::calculate_packet_size(
                descriptor.ancillary_data_header.user_data_words_count);
    }

    buffer.assign(RTP_ST_2110_40_ANCILLARY_HEADER_SIZE + payload_bytes, 0);

    RTP_SMPTE_2110_40_PacketContext writer_context;
    writer_context.version = 2;
    writer_context.sequence = sequence;
    writer_context.timestamp = timestamp;
    writer_context.ssrc = ssrc;
    writer_context.extended_sequence_number = extended_sequence;
    writer_context.field_indicator = field_indicator;
    writer_context.payload_size = payload_bytes;
    writer_context.payload_ptr = const_cast<byte_t*>(user_data_words_bytes.data());
    // const_cast is safe: fill_header/fill_payload only read from descriptors
    writer_context.descriptors = const_cast<std::vector<AncillaryDataDescriptor>*>(&descriptors);
    writer_context.descriptor_start_index = 0;
    writer_context.descriptor_count_in_packet = descriptors.size();

    RTP_SMPTE_2110_40_PacketWriter writer(buffer.data(), nullptr);
    size_t header_size = 0;
    size_t payload_size = 0;
    ASSERT_EQ(writer.fill_header(writer_context, header_size, nullptr), ReturnStatus::success);
    ASSERT_EQ(writer.fill_payload(writer_context, payload_size, nullptr), ReturnStatus::success);
}

#endif /* RDK_UTESTS_SERVICES_ULP_PACKET_RTP_SMPTE_2110_40_TEST_UTILS_H_ */
