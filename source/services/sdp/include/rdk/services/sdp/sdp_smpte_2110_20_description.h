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

#ifndef RDK_SERVICES_SDP_SDP_SMPTE_2110_20_DESCRIPTION_H_
#define RDK_SERVICES_SDP_SDP_SMPTE_2110_20_DESCRIPTION_H_

#include <vector>

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
 * @brief: SDP Media description builder for video streams following SMPTE ST 2110-10/20/21:2021.
 *
 * This class is responsible for constructing the media description part of the SDP string for video streams.
 * It is based on the SMPTE ST 2110-10/20/21:2021 specifications.
 *
 * The following media description attributes are supported:
 * --------------------------------------------------------------------------------------------------------
 *     - Source Filter Attribute:
 *           a=source-filter: <filter-mode> <filter-spec>(sub-components:<nettype> <address-types>
 *             <dest-address> <src-list>)
 * --------------------------------------------------------------------------------------------------------
 *     - RTP Map Attribute:
 *           a=rtpmap:<payload type> raw/90000
 * --------------------------------------------------------------------------------------------------------
 *     - Media Format Attribute:
 *           a=fmtp:<format> sampling=<sampling>; depth=<depth>; width=<width>; height=<height>;
 *             exactframerate=<exactframerate>; colorimetry=<colorimetry>; PM=<PM>; SSN=<SSN>;
 *             TP=<TP> [optional parameters: interlace / CMAX=<CMAX> / MAXUDP=<MAXUDP> / TSMODE=<TSMODE>
 *             / TSDELAY=<TSDELAY>]
 * --------------------------------------------------------------------------------------------------------
 *    - Timestamp Reference Clock Signalling Attribute:
 *           a=ts-refclk:[ptp=IEEE1588-2008:[[<grandmaster-clock-identity>:<domain-number> | traceable]] |
 *                       localmac=<mac-address>]
 * --------------------------------------------------------------------------------------------------------
 *    - Media Clock Signalling Attribute:
 *           a=mediaclk:[direct=0 | sender]
 */
class SMPTE2110_20_MediaDescription : public BaseMediaDescription
{
public:
    ~SMPTE2110_20_MediaDescription() = default;
    std::vector<json> get_media_description_attributes() const override;
    /**
     * @brief: Builder class for constructing SMPTE2110_20_MediaDescriptionAttributeBuilder objects.
     */
    class Builder : public BaseMediaDescription::Builder<Builder, SMPTE2110_20_MediaDescription>
    {
    public:
        using BaseMediaDescription::Builder<Builder, SMPTE2110_20_MediaDescription>::Builder;
        /**
         * @brief: Constructor for mandatory parameters.
         *
         * @param [in] transport_port: The transport port.
         * @param [in] transport_protocol: The transport protocol.
         * @param [in] media_format_description: The media format description.
         * @param [in] connection_address: The connection address.
         */
        explicit Builder(
            size_t transport_port,
            TransportProtocol transport_protocol,
            const std::string& media_format_description,
            const std::string& connection_address)
            : BaseMediaDescription::Builder<Builder, SMPTE2110_20_MediaDescription>(
                  MediaType::Video, transport_port, transport_protocol, media_format_description, connection_address) {}

        // Setters for optional parameters:

        /**
         * @brief: Sets the source filter attribute.
         *
         * This corresponds to the "a=source-filter" attribute in SDP as per RFC4570.
         *
         * @param [in] source_filter: The source filter attribute.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_source_filter(std::shared_ptr<SourceFilterAttribute> source_filter)
        {
            return set(m_instance->m_source_filter, source_filter);
        }
        /**
         * @brief: Sets the payload type.
         *
         * This corresponds to the <payload type> field in "a=rtpmap" attribute in SDP as per RFC4566.
         *
         * @param [in] payload_type: The payload type.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_payload_type(size_t payload_type) { return set(m_instance->m_payload_type, payload_type); }
        /**
         * @brief: Sets the media format.
         *
         * This corresponds to the <format> field in "a=fmtp" attribute in SDP as per RFC4566.
         *
         * @param [in] media_format: The media format.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_media_format(size_t media_format) { return set(m_instance->m_media_format, media_format); }
        /**
         * @brief: Sets the video sampling.
         *
         * This corresponds to the <sampling> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-20.
         *
         * @param [in] sampling: The sampling.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_sampling(VideoSampling sampling) { return set(m_instance->m_sampling, sampling); }
        /**
         * @brief: Sets the depth.
         *
         * This corresponds to the <depth> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-20.
         *
         * @param [in] depth: The depth.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_depth(ColorBitDepth depth) { return set(m_instance->m_depth, depth); }
        /**
         * @brief: Sets the width.
         *
         * This corresponds to the <width> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-20.
         *
         * @param [in] width: The width.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_width(size_t width) { return set(m_instance->m_width, width); }
        /**
         * @brief: Sets the height.
         *
         * This corresponds to the <height> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-20.
         *
         * @param [in] height: The height.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_height(size_t height) { return set(m_instance->m_height, height); }
        /**
         * @brief: Sets the exact frame rate.
         *
         * This corresponds to the <exactframerate> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-20.
         *
         * @param [in] exact_frame_rate: The exact frame rate.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_exact_frame_rate(const std::string& exact_frame_rate)
        {
            return set(m_instance->m_exact_frame_rate, exact_frame_rate);
        }
        /**
         * @brief: Sets the colorimetry.
         *
         * This corresponds to the <colorimetry> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-20.
         *
         * @param [in] colorimetry: The colorimetry.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_colorimetry(Colorimetry colorimetry) { return set(m_instance->m_colorimetry, colorimetry); }
        /**
         * @brief: Sets the packaging mode.
         *
         * This corresponds to the <PM> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-20.
         *
         * @param [in] packaging_mode: The packaging mode.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_packaging_mode(PackagingMode packaging_mode)
        {
            return set(m_instance->m_packaging_mode, packaging_mode);
        }
        /**
         * @brief: Sets the SMPTE standard number.
         *
         * This corresponds to the <SSN> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-20.
         *
         * @param [in] smpte_standard_number: The SMPTE standard number.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_smpte_standard_number(SMPTEStandardNumber smpte_standard_number)
        {
            return set(m_instance->m_smpte_standard_number, smpte_standard_number);
        }
        /**
         * @brief: Sets the sender type.
         *
         * This corresponds to the <TP> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-21.
         *
         * @param [in] sender_type: The sender type.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_sender_type(SenderType sender_type) { return set(m_instance->m_sender_type, sender_type); }
        /**
         * @brief: Sets the video scan type.
         *
         * This corresponds to the <interlace> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-20.
         *
         * @param [in] video_scan_type: The video scan type.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_video_scan_type(VideoScanType video_scan_type)
        {
            return set(m_instance->m_video_scan_type, video_scan_type);
        }
        /**
         * @brief: Sets the CMAX value.
         *
         * This corresponds to the <CMAX> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-21.
         *
         * @param [in] cmax: The CMAX value.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_cmax(size_t cmax) { return set(m_instance->m_cmax, cmax); }
        /**
         * @brief: Sets the maximum UDP packet size.
         *
         * This corresponds to the <MAXUDP> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-20.
         *
         * @param [in] max_udp: The maximum UDP packet size.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_max_udp(size_t max_udp) { return set(m_instance->m_max_udp, max_udp); }
        /**
         * @brief: Sets the timestamp mode.
         *
         * This corresponds to the <TSMODE> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-10.
         *
         * @param [in] timestamp_mode: The timestamp mode.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_timestamp_mode(TimestampMode timestamp_mode)
        {
            return set(m_instance->m_timestamp_mode, timestamp_mode);
        }
        /**
         * @brief: Sets the timestamp delay.
         *
         * This corresponds to the <TSDELAY> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-10.
         *
         * @param [in] timestamp_delay: The timestamp delay.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_timestamp_delay(size_t timestamp_delay)
        {
            return set(m_instance->m_timestamp_delay, timestamp_delay);
        }
        /**
         * @brief: Sets the timestamp reference clock.
         *
         * This corresponds to the <ts-refclk> field in "a=ts-refclk" attribute in SDP as per SMPTE ST 2110-10.
         *
         * @param [in] timestamp_ref_clock: The timestamp reference clock.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_timestamp_ref_clock(TimestampRefClock timestamp_ref_clock)
        {
            return set(m_instance->m_timestamp_ref_clock, timestamp_ref_clock);
        }
        /**
         * @brief: Sets the PTP grandmaster clock identity for the timestamp reference clock.
         *
         * This corresponds to the <grandmaster-clock-identity> field in "a=ts-refclk" attribute in SDP as per SMPTE ST
         * 2110-10.
         *
         * @param [in] timestamp_ref_clock_ptp_grandmaster_clock_identity: The PTP grandmaster clock identity.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_timestamp_ref_clock_ptp_grandmaster_clock_identity(
            const std::string& timestamp_ref_clock_ptp_grandmaster_clock_identity)
        {
            return set(
                m_instance->m_timestamp_ref_clock_ptp_grandmaster_clock_identity,
                timestamp_ref_clock_ptp_grandmaster_clock_identity
            );
        }
        /**
         * @brief: Sets the PTP domain number for the timestamp reference clock.
         *
         * This corresponds to the <domain-number> field in "a=ts-refclk" attribute in SDP as per SMPTE ST 2110-10.
         *
         * @param [in] timestamp_ref_clock_ptp_domain_number: The PTP domain number.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_timestamp_ref_clock_ptp_domain_number(size_t timestamp_ref_clock_ptp_domain_number)
        {
            return set(m_instance->m_timestamp_ref_clock_ptp_domain_number, timestamp_ref_clock_ptp_domain_number);
        }
        /**
         * @brief: Sets the traceable flag for the PTP timestamp reference clock.
         *
         * This corresponds to the <traceable> field in "a=ts-refclk" attribute in SDP as per SMPTE ST 2110-10.
         *
         * @param [in] timestamp_ref_clock_ptp_traceable: The traceable flag.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_timestamp_ref_clock_ptp_traceable(bool timestamp_ref_clock_ptp_traceable)
        {
            return set(m_instance->m_timestamp_ref_clock_ptp_traceable, timestamp_ref_clock_ptp_traceable);
        }
        /**
         * @brief: Sets the local MAC address for the timestamp reference clock.
         *
         * This corresponds to the <localmac> field in "a=ts-refclk" attribute in SDP as per SMPTE ST 2110-10.
         *
         * @param [in] timestamp_ref_clock_local_mac: The local MAC address.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_timestamp_ref_clock_local_mac(const std::string& timestamp_ref_clock_local_mac)
        {
            return set(m_instance->m_timestamp_ref_clock_local_mac, timestamp_ref_clock_local_mac);
        }
        /**
         * @brief: Sets the media clock.
         *
         * This corresponds to the <mediaclk> field in "a=mediaclk" attribute in SDP as per SMPTE ST 2110-10.
         *
         * @param [in] media_clock: The media clock.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_media_clock(MediaClock media_clock) { return set(m_instance->m_media_clock, media_clock); }
        /**
         * @brief: Sets the extra format specific parameters.
         *
         * This corresponds to the <format specific parameters> field in "a=fmtp" attribute in SDP as per SMPTE ST 2110-20.
         * It allows to add extra format specific parameters to the media description.
         *
         * @param [in] extra_format_specific_parameters: The extra format specific parameters.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_extra_format_specific_parameters(
            const std::vector<FormatSpecificParameter>& extra_format_specific_parameters)
        {
            return set(m_instance->m_extra_format_specific_parameters, extra_format_specific_parameters);
        }
    };

private:
    /**
     * @brief: Default constructor for SMPTE2110_20_MediaDescription.
     *
     * This constructor is private and only accessible by the Builder class.
     */
    SMPTE2110_20_MediaDescription() = default;

    std::shared_ptr<SourceFilterAttribute> m_source_filter = nullptr;
    size_t m_payload_type = 96;
    size_t m_media_format = 96;
    VideoSampling m_sampling = VideoSampling::YCbCr_4_2_2;
    ColorBitDepth m_depth = ColorBitDepth::_10;
    size_t m_width = 1920;
    size_t m_height = 1080;
    std::string m_exact_frame_rate = "60";
    Colorimetry m_colorimetry = Colorimetry::BT709;
    PackagingMode m_packaging_mode = PackagingMode::_2110GPM;
    SMPTEStandardNumber m_smpte_standard_number = SMPTEStandardNumber::ST2110_20_2017;
    SenderType m_sender_type = SenderType::_2110TPN;
    VideoScanType m_video_scan_type = VideoScanType::Progressive;
    size_t m_cmax = 0;
    size_t m_max_udp = 0;
    TimestampMode m_timestamp_mode = TimestampMode::Unknown;
    size_t m_timestamp_delay = 0;
    TimestampRefClock m_timestamp_ref_clock = TimestampRefClock::PTP;
    std::string m_timestamp_ref_clock_ptp_grandmaster_clock_identity = "";
    size_t m_timestamp_ref_clock_ptp_domain_number = 0;
    bool m_timestamp_ref_clock_ptp_traceable = true;
    std::string m_timestamp_ref_clock_local_mac = "";
    MediaClock m_media_clock = MediaClock::Direct;
    std::vector<FormatSpecificParameter> m_extra_format_specific_parameters;

    friend class ISDP::IBuilder<SMPTE2110_20_MediaDescription, Builder>;
    friend class BaseMediaDescription::Builder<Builder, SMPTE2110_20_MediaDescription>;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_SDP_SDP_SMPTE_2110_20_DESCRIPTION_H_ */
