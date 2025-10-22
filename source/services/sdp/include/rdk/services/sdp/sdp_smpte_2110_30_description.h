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

#ifndef RDK_SERVICES_SDP_SDP_SMPTE_2110_30_DESCRIPTION_H_
#define RDK_SERVICES_SDP_SDP_SMPTE_2110_30_DESCRIPTION_H_

#include <vector>
#include <initializer_list>

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
enum class ChannelGroupingSymbol;

/**
 * @brief: SDP Media description builder for audio streams following SMPTE ST 2110-30.
 *
 * This class is responsible for constructing the media description part of the SDP string for audio streams.
 * It is based on the SMPTE ST 2110-30 and ST 2110-10 specifications.
 *
 * The following media description attributes are supported:
 * --------------------------------------------------------------------------------------------------------
 *     - Source Filter Attribute:
 *           a=source-filter: <filter-mode> <filter-spec>(sub-components:<nettype> <address-types>
 *             <dest-address> <src-list>)
 * --------------------------------------------------------------------------------------------------------
 *     - RTP Map Attribute:
 *           a=rtpmap:<payload type> <encoding>/<clock rate>/<channels>
 * --------------------------------------------------------------------------------------------------------
 *     - Media Format Attribute:
 *           a=fmtp:<format> channel-order=<channel-order> [optional parameters]
 * --------------------------------------------------------------------------------------------------------
 *     - Packet Time Attributes:
 *           a=ptime:<packet-time-ms>
 *           a=maxptime:<max-packet-time-ms>
 * --------------------------------------------------------------------------------------------------------
 *     - Timestamp Reference Clock Signalling Attribute:
 *           a=ts-refclk:[ptp=IEEE1588-2008:[[<grandmaster-clock-identity>:<domain-number> | traceable]] |
 *                       localmac=<mac-address>]
 * --------------------------------------------------------------------------------------------------------
 *     - Media Clock Signalling Attribute:
 *           a=mediaclk:[direct=0 | sender]
 */
class SMPTE2110_30_MediaDescription : public BaseMediaDescription
{
public:
    ~SMPTE2110_30_MediaDescription() = default;
    std::vector<json> get_media_description_attributes() const override;

    class Builder : public BaseMediaDescription::Builder<Builder, SMPTE2110_30_MediaDescription>
    {
    public:
        using BaseMediaDescription::Builder<Builder, SMPTE2110_30_MediaDescription>::Builder;

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
            : BaseMediaDescription::Builder<Builder, SMPTE2110_30_MediaDescription>(
                  MediaType::Audio, transport_port, transport_protocol, media_format_description, connection_address)
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
         * @brief: Sets the audio encoding name.
         *
         * This corresponds to the <encoding name> in "a=rtpmap" attribute in SDP as per SMPTE ST 2110-30.
         *
         * @param [in] encoding: The audio encoding (e.g., L24).
         *
         * @return: Reference to the builder object.
         */
        Builder& set_encoding(AudioEncoding encoding) { return set(m_instance->m_encoding, encoding); }
        /**
         * @brief: Sets the sampling rate.
         *
         * This corresponds to the <clock rate> in "a=rtpmap" attribute in SDP as per SMPTE ST 2110-30.
         *
         * @param [in] sampling_rate: The audio sampling rate.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_sampling_rate(AudioSamplingRate sampling_rate) {
            return set(m_instance->m_sampling_rate, sampling_rate);
        }
        /**
         * @brief: Sets the number of audio channels.
         *
         * This corresponds to the <channels> in "a=rtpmap" attribute in SDP as per SMPTE ST 2110-30.
         *
         * @param [in] channels: The number of audio channels.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_channels(size_t channels) { return set(m_instance->m_channels, channels); }
        /**
         * @brief: Sets the channel order descriptor.
         *
         * This corresponds to the channel-order parameter in "a=fmtp" attribute in SDP as per SMPTE ST 2110-30.
         */
        Builder& set_channel_order(std::initializer_list<ChannelGroupingSymbol> symbols);
        /**
         * @brief: Sets the packet time.
         *
         * This corresponds to the <ptime> attribute in SDP as per RFC4566.
         *
         * @param [in] ptime_ms: The packet time in milliseconds.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_ptime(double ptime_ms) { return set(m_instance->m_ptime_ms, ptime_ms); }
        /**
         * @brief: Sets the maximum packet time.
         *
         * This corresponds to the <maxptime> attribute in SDP as per RFC4566.
         *
         * @param [in] maxptime_ms: The maximum packet time in milliseconds.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_maxptime(double maxptime_ms) { return set(m_instance->m_maxptime_ms, maxptime_ms); }
    };

private:
    SMPTE2110_30_MediaDescription() = default;

    std::shared_ptr<SourceFilterAttribute> m_source_filter = nullptr;
    uint8_t m_payload_type = PAYLOAD_TYPE_ST_2110_30;
    uint8_t m_media_format = MEDIA_FORMAT_ST_2110_30;
    AudioEncoding m_encoding = AudioEncoding::L24;
    AudioSamplingRate m_sampling_rate = AudioSamplingRate::_48000;
    size_t m_channels = 2;
    std::string m_channel_order_values = "";
    TimestampRefClock m_timestamp_ref_clock = TimestampRefClock::PTP;
    std::string m_timestamp_ref_clock_ptp_grandmaster_clock_identity = "";
    size_t m_timestamp_ref_clock_ptp_domain_number = 0;
    bool m_timestamp_ref_clock_ptp_traceable = true;
    std::string m_timestamp_ref_clock_local_mac = "";
    MediaClock m_media_clock = MediaClock::Direct;
    std::vector<FormatSpecificParameter> m_extra_format_specific_parameters;
    double m_ptime_ms = 1.0;
    double m_maxptime_ms = 1.0;

    friend class ISDP::IBuilder<SMPTE2110_30_MediaDescription, Builder>;
    friend class BaseMediaDescription::Builder<Builder, SMPTE2110_30_MediaDescription>;
};

} /* namespace services */
} /* namespace dev_kit */
} /* namespace rivermax */

#endif /* RDK_SERVICES_SDP_SDP_SMPTE_2110_30_DESCRIPTION_H_ */
