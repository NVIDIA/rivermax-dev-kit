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

#ifndef RDK_SERVICES_SDP_SDP_SMPTE_2110_40_DESCRIPTION_H_
#define RDK_SERVICES_SDP_SDP_SMPTE_2110_40_DESCRIPTION_H_

#include <vector>
#include <cstdint>

#include "sdptransform.hpp"

#include "rdk/services/sdp/sdp_defs.h"
#include "rdk/services/sdp/sdp_interface.h"
#include "rdk/services/sdp/sdp_common_descriptions.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: SDP Media description builder for ancillary data streams following SMPTE ST 2110-40.
 *
 * This class is responsible for constructing the media description part of the SDP string for ancillary
 * data streams. It is based on the SMPTE ST 2110-40 and ST 2110-10 specifications.
 *
 * The following media description attributes are supported:
 * --------------------------------------------------------------------------------------------------------
 *     - Source Filter Attribute:
 *           a=source-filter: <filter-mode> <filter-spec>(sub-components:<nettype> <address-types>
 *             <dest-address> <src-list>)
 * --------------------------------------------------------------------------------------------------------
 *     - RTP Map Attribute:
 *           a=rtpmap:<payload type> smpte291/90000
 * --------------------------------------------------------------------------------------------------------
 *     - Media Format Attribute:
 *           a=fmtp:<format> [DID_SDID={<DID>,<SDID>}; ...] [VPID_Code=<code>] [optional parameters]
 *           - DID_SDID parameters follow RFC 8331 and may appear multiple times on the same line
 *           - VPID_Code values map to SMPTE ST 352 payload ID codes
 * --------------------------------------------------------------------------------------------------------
 *     - Timestamp Reference Clock Signalling Attribute:
 *           a=ts-refclk:[ptp=IEEE1588-2008:[[<grandmaster-clock-identity>:<domain-number> | traceable]] |
 *                       localmac=<mac-address>]
 * --------------------------------------------------------------------------------------------------------
 *     - Media Clock Signalling Attribute:
 *           a=mediaclk:[direct=0 | sender]
 */
class SMPTE2110_40_MediaDescription : public BaseMediaDescription
{
public:
    ~SMPTE2110_40_MediaDescription() = default;
    std::vector<json> get_media_description_attributes() const override;

    /**
     * @brief: Builder class for constructing SMPTE2110_40_MediaDescription objects.
     */
    class Builder : public BaseMediaDescription::Builder<Builder, SMPTE2110_40_MediaDescription>
    {
    public:
        using BaseMediaDescription::Builder<Builder, SMPTE2110_40_MediaDescription>::Builder;

        /**
         * @brief: Constructor for mandatory parameters.
         *
         * @param [in] transport_port: The transport port.
         * @param [in] transport_protocol: The transport protocol.
         * @param [in] media_format_description: The media format description (payload type as string).
         * @param [in] connection_address: The connection address.
         */
        explicit Builder(
            size_t transport_port,
            TransportProtocol transport_protocol,
            const std::string& media_format_description,
            const std::string& connection_address)
            : BaseMediaDescription::Builder<Builder, SMPTE2110_40_MediaDescription>(
                  MediaType::Video, transport_port, transport_protocol, media_format_description, connection_address)
        {
            size_t payload_type_parsed = std::stoul(media_format_description);
            if (payload_type_parsed > RTP_PAYLOAD_TYPE_MAX) {
                throw "Invalid payload type";
            }
            uint8_t payload_type = static_cast<uint8_t>(payload_type_parsed);
            m_instance->m_payload_type = payload_type;
            m_instance->m_media_format = payload_type;
        }

        // Setters for optional parameters:

        /**
         * @brief: Sets a single DID/SDID pair.
         *
         * This corresponds to DID_SDID parameter in the "a=fmtp" attribute as per RFC 8331.
         *
         * @param [in] did: The DID byte (0x00..0xFF).
         * @param [in] sdid: The SDID byte (0x00..0xFF).
         *
         * @return: Reference to the builder object.
         */
        Builder& set_did_sdid(uint8_t did, uint8_t sdid) {
            m_instance->m_did_sdids.clear();
            m_instance->m_did_sdids.push_back({did, sdid});
            return static_cast<Builder&>(*this);
        }
        /**
         * @brief: Adds a DID/SDID pair.
         *
         * This corresponds to DID_SDID parameter in the "a=fmtp" attribute as per RFC 8331.
         *
         * @param [in] did: The DID byte (0x00..0xFF).
         * @param [in] sdid: The SDID byte (0x00..0xFF).
         *
         * @return: Reference to the builder object.
         */
        Builder& add_did_sdid(uint8_t did, uint8_t sdid) {
            m_instance->m_did_sdids.push_back({did, sdid});
            return static_cast<Builder&>(*this);
        }
        /**
         * @brief: Sets the VPID code.
         *
         * This corresponds to the VPID_Code parameter in the "a=fmtp" attribute as per SMPTE ST 2110-40 and SMPTE ST 352.
         *
         * @param [in] vpid_code: The VPID code value.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_vpid_code(uint8_t vpid_code) {
            m_instance->m_vpid_code = vpid_code; m_instance->m_vpid_code_set = true;
            return static_cast<Builder&>(*this);
        }
    };

private:
    /**
     * @brief: Default constructor for SMPTE2110_40_MediaDescription.
     *
     * This constructor is private and only accessible by the Builder class.
     */
    SMPTE2110_40_MediaDescription() = default;

    std::shared_ptr<SourceFilterAttribute> m_source_filter = nullptr;
    uint8_t m_payload_type = PAYLOAD_TYPE_ST_2110_40;
    uint8_t m_media_format = MEDIA_FORMAT_ST_2110_40;
    struct DidSdid { uint8_t did; uint8_t sdid; };
    std::vector<DidSdid> m_did_sdids;
    uint8_t m_vpid_code = 0;
    bool m_vpid_code_set = false;
    TimestampRefClock m_timestamp_ref_clock = TimestampRefClock::PTP;
    std::string m_timestamp_ref_clock_ptp_grandmaster_clock_identity = "";
    size_t m_timestamp_ref_clock_ptp_domain_number = 0;
    bool m_timestamp_ref_clock_ptp_traceable = true;
    std::string m_timestamp_ref_clock_local_mac = "";
    MediaClock m_media_clock = MediaClock::Direct;
    std::vector<FormatSpecificParameter> m_extra_format_specific_parameters;

    friend class ISDP::IBuilder<SMPTE2110_40_MediaDescription, Builder>;
    friend class BaseMediaDescription::Builder<Builder, SMPTE2110_40_MediaDescription>;
};

} /* namespace services */
} /* namespace dev_kit */
} /* namespace rivermax */

#endif /* RDK_SERVICES_SDP_SDP_SMPTE_2110_40_DESCRIPTION_H_ */
