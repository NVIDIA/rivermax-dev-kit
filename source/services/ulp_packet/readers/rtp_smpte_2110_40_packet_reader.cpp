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

#include "rdk/services/ulp_packet/readers/rtp_smpte_2110_40_packet_reader.h"
#include "rdk/services/ulp_packet/network_byte_order.h"
#include "rdk/services/ulp_packet/rtp_smpte_2110_40_extension.h"
#include "rdk/services/ulp_packet/smpte_2110_40_ancillary_utils.h"

#include <cstring>
#include <iostream>
#include <vector>

using namespace rdk::services;

size_t AncillaryDataPacketReader::read_ancillary_data(const byte_t* buffer,
    size_t remaining_bytes,
    size_t current_output_offset,
    AncillaryDataDescriptor& ancillary_data_descriptor,
    byte_t* output_ptr)
{
    const size_t min_packet_size = AncillaryDataUtils::calculate_packet_size(0);
    if (remaining_bytes < min_packet_size) {
        return 0;
    }

    // Parse the 4-byte fixed binary header (C, line, H_offset, S, stream).
    uint32_t fixed_header_raw;
    memcpy(&fixed_header_raw, buffer, sizeof(fixed_header_raw));
    const uint32_t fixed_header = ntohl(fixed_header_raw);

    auto& ancillary_data_header = ancillary_data_descriptor.ancillary_data_header;
    ancillary_data_header.c_flag = static_cast<bool>((fixed_header >> RTP_ST_2110_40_C_FLAG_BIT_POSITION) & 1);
    ancillary_data_header.line_number = static_cast<uint16_t>((fixed_header >> RTP_ST_2110_40_LINE_NUMBER_BIT_POSITION) & MASK_11BIT);
    ancillary_data_header.horizontal_offset = static_cast<uint16_t>((fixed_header >> RTP_ST_2110_40_HORIZONTAL_OFFSET_BIT_POSITION) & MASK_12BIT);
    ancillary_data_header.s_flag = static_cast<bool>((fixed_header >> RTP_ST_2110_40_S_FLAG_BIT_POSITION) & 1);
    ancillary_data_header.stream_number = static_cast<uint8_t>(fixed_header & MASK_7BIT);

    // Unpack and validate DID, SDID, DATA_COUNT (the three fixed 10-bit words).
    const uint8_t* packed_words_start = reinterpret_cast<const uint8_t*>(buffer)
        + RTP_ST_2110_40_FIXED_DATA_HEADER_SIZE;

    constexpr size_t FIXED_WORD_COUNT = RTP_ST_2110_40_DID_WORD_COUNT
        + RTP_ST_2110_40_SDID_WORD_COUNT
        + RTP_ST_2110_40_DATA_COUNT_WORD_COUNT;

    std::vector<uint16_t> data_words(FIXED_WORD_COUNT);
    AncillaryDataUtils::unpack_10bit_words(packed_words_start, 0, FIXED_WORD_COUNT, data_words.data());

    for (size_t i = 0; i < FIXED_WORD_COUNT; ++i) {
        const bool parity_valid = AncillaryDataUtils::verify_parity_bits(data_words[i]);
        if (!parity_valid) {
            return 0;
        }
    }

    ancillary_data_header.did = AncillaryDataUtils::strip_parity_bits(data_words[0]);
    ancillary_data_header.sdid = AncillaryDataUtils::strip_parity_bits(data_words[1]);
    ancillary_data_header.user_data_words_count = AncillaryDataUtils::strip_parity_bits(data_words[2]);

    const uint16_t user_data_words_count = ancillary_data_header.user_data_words_count;

    const size_t sub_packet_size = AncillaryDataUtils::calculate_packet_size(user_data_words_count);
    if (remaining_bytes < sub_packet_size) {
        return 0;
    }

    // Extend data_words to hold DID+SDID+DATA_COUNT+UDW, unpack UDW into the tail,
    // then verify parity and strip to output buffer.
    ancillary_data_descriptor.user_data_offset = current_output_offset;
    data_words.resize(FIXED_WORD_COUNT + user_data_words_count);
    AncillaryDataUtils::unpack_10bit_words(packed_words_start, FIXED_WORD_COUNT,
        user_data_words_count, data_words.data() + FIXED_WORD_COUNT);

    for (uint16_t i = 0; i < user_data_words_count; ++i) {
        const bool parity_valid = AncillaryDataUtils::verify_parity_bits(data_words[FIXED_WORD_COUNT + i]);
        if (!parity_valid) {
            return 0;
        }
        output_ptr[i] = AncillaryDataUtils::strip_parity_bits(data_words[FIXED_WORD_COUNT + i]);
    }

    // Validate the checksum word.
    uint16_t checksum_word;
    AncillaryDataUtils::unpack_10bit_words(packed_words_start,
        FIXED_WORD_COUNT + user_data_words_count, 1, &checksum_word);
    const uint16_t calculated_checksum = AncillaryDataUtils::calculate_checksum(data_words.data(), data_words.size());
    if (checksum_word != calculated_checksum) {
        return 0;
    }

    return sub_packet_size;
}

ReturnStatus RTP_SMPTE_2110_40_PacketReader::parse_header(IPacketContext& context)
{
    ReturnStatus status = RTPPacketReader::parse_header(context);
    if (status != ReturnStatus::success) {
        std::cerr << "Failed to parse RTP header for ST 2110-40 ancillary packet" << std::endl;
        return status;
    }

    auto& rtp_packet_context = static_cast<RTP_SMPTE_2110_40_PacketContext&>(context);

    const AncillaryRTPExtension* ancillary_header_extension_ptr =
        reinterpret_cast<const AncillaryRTPExtension*>(m_header_ptr + RTPPacketReader::get_header_size());

    const uint16_t ext_seq_high = ntohs(ancillary_header_extension_ptr->extended_sequence_number);
    rtp_packet_context.extended_sequence_number =
        (static_cast<uint32_t>(ext_seq_high) << 16) | rtp_packet_context.sequence;
    rtp_packet_context.length = ntohs(ancillary_header_extension_ptr->length);
    rtp_packet_context.ancillary_count = static_cast<uint32_t>(ancillary_header_extension_ptr->anc_count);
    rtp_packet_context.field_indicator = ancillary_header_extension_ptr->get_field_indicator();

    m_payload_size = rtp_packet_context.length;

    return ReturnStatus::success;
}

ReturnStatus RTP_SMPTE_2110_40_PacketReader::parse_payload(IPacketContext& context)
{
    auto& rtp_packet_context = static_cast<RTP_SMPTE_2110_40_PacketContext&>(context);

    rtp_packet_context.descriptor_start_index = rtp_packet_context.descriptors ? rtp_packet_context.descriptors->size() : 0;
    rtp_packet_context.descriptor_count_in_packet = 0;

    if (rtp_packet_context.descriptors == nullptr || rtp_packet_context.payload_ptr == nullptr) {
        std::cerr << "Error: ST 2110-40 payload parse context missing output buffer or descriptor list"
            << std::endl;
        return ReturnStatus::failure;
    }

    if (rtp_packet_context.ancillary_count == 0) {
        return ReturnStatus::success;
    }

    const byte_t* payload_start = (m_payload_ptr != nullptr)
        ? m_payload_ptr
        : (m_header_ptr + get_header_size());

    const byte_t* current_ptr = payload_start;
    const byte_t* payload_end = payload_start + rtp_packet_context.length;
    byte_t* output_ptr = rtp_packet_context.payload_ptr;
    size_t write_offset = 0;

    // Iterate over the expected number of ANC sub-packets declared in the RTP extension header.
    for (uint32_t i = 0; i < rtp_packet_context.ancillary_count; ++i) {
        if (current_ptr >= payload_end) {
            std::cerr << "Error: ST 2110-40 payload truncated at sub-packet " << i << std::endl;
            return ReturnStatus::failure;
        }

        AncillaryDataDescriptor ancillary_data_descriptor;
        const size_t remaining = static_cast<size_t>(payload_end - current_ptr);
        const size_t bytes_consumed = AncillaryDataPacketReader::read_ancillary_data(
            current_ptr, remaining, write_offset,
            ancillary_data_descriptor, output_ptr);
        if (bytes_consumed == 0) {
            std::cerr << "Error: read_ancillary_data failed at sub-packet " << i << std::endl;
            return ReturnStatus::failure;
        }

        const uint16_t user_data_words_count = ancillary_data_descriptor.ancillary_data_header.user_data_words_count;
        rtp_packet_context.descriptors->push_back(std::move(ancillary_data_descriptor));
        rtp_packet_context.descriptor_count_in_packet++;
        current_ptr += bytes_consumed;
        output_ptr += user_data_words_count;
        write_offset += user_data_words_count;
    }

    return ReturnStatus::success;
}
