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

#ifndef RDK_SERVICES_SDP_SDP_COMMON_DESCRIPTIONS_H_
#define RDK_SERVICES_SDP_SDP_COMMON_DESCRIPTIONS_H_

#include <memory>
#include <vector>

#include "sdptransform.hpp"

#include "rdk/services/sdp/sdp_defs.h"
#include "rdk/services/sdp/sdp_interface.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: SDP Session description builder.
 *
 * This class is responsible for constructing the session description part of the SDP string.
 * It is based on the RFC4566 specification.
 *
 * The following session description specifications are supported:
 * --------------------------------------------------------------------------------------------------------
 *     - Protocol Version:
 *           v=<version>
 * --------------------------------------------------------------------------------------------------------
 *     - Origin:
 *           o=<username> <sess-id> <sess-version> <nettype> <addrtype> <unicast-address>
 * --------------------------------------------------------------------------------------------------------
 *     - Session Name:
 *           s=<session name>
 * --------------------------------------------------------------------------------------------------------
 */
class SessionDescription : public ISDP
{
public:
    ~SessionDescription() = default;
    operator json() const override;
    /**
     * @brief: Builder class for constructing SessionDescription objects.
     */
    class Builder : public ISDP::IBuilder<SessionDescription, Builder>
    {
    public:
        /**
         * @brief: Constructor for mandatory parameters.
         *
         * @param [in] unicast_address: The unicast address.
         */
        explicit Builder(const std::string& unicast_address) : ISDP::IBuilder<SessionDescription, Builder>()
        {
            throw_if(unicast_address.empty());
            m_instance->m_unicast_address = unicast_address;
        }

        // Setters for optional parameters:

        /**
         * @brief: Sets the protocol version.
         *
         * This corresponds to the <version> field in "v=" line in SDP as per RFC4566.
         *
         * @param [in] protocol_version: The protocol version.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_protocol_version(size_t protocol_version)
        {
            return set(m_instance->m_protocol_version, protocol_version);
        }
        /**
         * @brief: Sets the username.
         *
         * This corresponds to the <username> field in "o=" line in SDP as per RFC4566.
         *
         * @param [in] username: The username.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_username(const std::string& username) { return set(m_instance->m_username, username); }
        /**
         * @brief: Sets the session ID.
         *
         * This corresponds to the <sess-id> field in "o=" line in SDP as per RFC4566.
         *
         * @param [in] session_id: The session ID.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_session_id(size_t session_id) { return set(m_instance->m_session_id, session_id); }
        /**
         * @brief: Sets the session version.
         *
         * This corresponds to the <sess-version> field in "o=" line in SDP as per RFC4566.
         *
         * @param [in] session_version: The session version.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_session_version(size_t session_version)
        {
            return set(m_instance->m_session_version, session_version);
        }
        /**
         * @brief: Sets the network type.
         *
         * This corresponds to the <nettype> field in "o=" line in SDP as per RFC4566.
         *
         * @param [in] network_type: The network type.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_network_type(NetworkType network_type) { return set(m_instance->m_network_type, network_type); }
        /**
         * @brief: Sets the address type.
         *
         * This corresponds to the <addrtype> field in "o=" line in SDP as per RFC4566.
         *
         * @param [in] address_type: The address type.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_address_type(AddressType address_type) { return set(m_instance->m_address_type, address_type); }
        /**
         * @brief: Sets the session name.
         *
         * This corresponds to the <session name> field in "s=" line in SDP as per RFC4566.
         *
         * @param [in] session_name: The session name.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_session_name(const std::string& session_name)
        {
            return set(m_instance->m_session_name, session_name);
        }
    };

private:
    /**
     * @brief: Default constructor for SessionDescription.
     *
     * This constructor is private and only accessible by the Builder class.
     */
    SessionDescription() = default;

    size_t m_protocol_version = 0;
    std::string m_username = "-";
    size_t m_session_id = 0;
    size_t m_session_version = 0;
    NetworkType m_network_type = NetworkType::_IN;
    AddressType m_address_type = AddressType::IP4;
    std::string m_unicast_address;
    std::string m_session_name;

    friend class ISDP::IBuilder<SessionDescription, Builder>;
};
/**
 * @brief: SDP Time description builder.
 *
 * This class is responsible for constructing the time description part of the SDP string.
 * It is based on the RFC4566 specification.
 *
 * The following session description specifications are supported:
 * --------------------------------------------------------------------------------------------------------
 *     - Timing:
 *           t=<start-time> <stop-time>
 * --------------------------------------------------------------------------------------------------------
 */
class TimeDescription : public ISDP
{
public:
    ~TimeDescription() = default;
    operator json() const override;
    /**
     * @brief: Builder class for constructing TimeDescription objects.
     */
    class Builder : public ISDP::IBuilder<TimeDescription, Builder>
    {
    public:
        /**
         * @brief: Constructor for mandatory parameters.
         */
        explicit Builder() : ISDP::IBuilder<TimeDescription, Builder>() {}

        // Setters for optional parameters:

        /**
         * @brief: Sets the start time.
         *
         * This corresponds to the <start-time> field in "t=" line in SDP as per RFC4566.
         *
         * @param [in] start_time: The start time.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_start_time(size_t start_time) { return set(m_instance->m_start_time, start_time); }
        /**
         * @brief: Sets the stop time.
         *
         * This corresponds to the <stop-time> field in "t=" line in SDP as per RFC4566.
         *
         * @param [in] stop_time: The stop time.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_stop_time(size_t stop_time) { return set(m_instance->m_stop_time, stop_time); }
    };

private:
    /**
     * @brief: Default constructor for TimeDescription.
     *
     * This constructor is private and only accessible by the Builder class.
     */
    TimeDescription() = default;

    size_t m_start_time = 0;
    size_t m_stop_time = 0;

    friend class ISDP::IBuilder<TimeDescription, Builder>;
};
/**
 * @brief: Media format specific parameter.
 *
 * Defines the <format specific parameters> of Media Format Attribute:
 *     a=fmtp:<format> <format specific parameters> [optional parameters]
 *
 * @param [in] name: The format specific optional parameter name.
 * @param [in] value: The format specific optional parameter value.
 * @param [in] condition: The condition based on which the optional parameter is included.
 */
struct FormatSpecificParameter
{
    std::string name;
    std::string value;
    bool condition = false;
};
/**
 * @brief: Media format attribute.
 *
 * Defines the Media Format Attribute:
 *     a=fmtp:<format> <format specific parameters>
 *
 * @param [in] format: The media format.
 * @param [in] parameters: The format specific parameters.
 */
struct MediaFormatAttribute
{
    size_t format;
    std::vector<FormatSpecificParameter> parameters;
};
/**
 * @brief: RTP map attribute.
 *
 * Defines the RTP Map Attribute:
 *     a=rtpmap:<payload type> <encoding name>/<clock rate>[/<encoding parameters>]
 *
 * @param [in] payload_type: The payload type.
 * @param [in] encoding_name: The encoding name.
 * @param [in] clock_rate: The clock rate.
 * @param [in] encoding_parameters: The encoding parameters (optional).
 */
struct RTPMapAttribute
{
    size_t payload_type;
    std::string encoding_name;
    size_t clock_rate;
    std::string encoding_parameters;
};
/**
 * @brief: Source filter attribute.
 *
 * This class is responsible for constructing the source filter attribute part of the SDP string.
 * It is based on the RFC4570 specification.
 *
 * Defines the source filter attribute:
 *     a=source-filter: <filter-mode> <filter-spec>(sub-components:<nettype> <address-types>
 *       <dest-address> <src-list>)
 */
class SourceFilterAttribute : public ISDP
{
public:
    ~SourceFilterAttribute() = default;
    operator json() const override;
    /**
     * @brief: Builder class for constructing SourceFilterAttribute objects.
     */
    class Builder : public ISDP::IBuilder<SourceFilterAttribute, Builder>
    {
    public:
        /**
         * @brief: Constructor for mandatory parameters.
         *
         * @param [in] destination_address: The destination address, corresponds to the <dest-address> field.
         * @param [in] source_list: The source list, corresponds to the <src-list> field.
         */
        explicit Builder(const std::string& destination_address, const std::string& source_list)
            : ISDP::IBuilder<SourceFilterAttribute, Builder>()
        {
            throw_if(destination_address.empty() || source_list.empty());
            m_instance->destination_address = destination_address;
            m_instance->source_list = source_list;
        }

        // Setters for optional parameters:

        /**
         * @brief: Sets the filter mode.
         *
         * This corresponds to the <filter-mode> field in "a=source-filter" attribute in SDP as per RFC4570.
         *
         * @param [in] filter_mode: The filter mode.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_filter_mode(NetworkFilterMode filter_mode) { return set(m_instance->filter_mode, filter_mode); }
        /**
         * @brief: Sets the network type.
         *
         * This corresponds to the <nettype> field in "a=source-filter" attribute in SDP as per RFC4570.
         *
         * @param [in] network_type: The network type.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_network_type(NetworkType network_type) { return set(m_instance->network_type, network_type); }
        /**
         * @brief: Sets the address type.
         *
         * This corresponds to the <addrtype> field in "a=source-filter" attribute in SDP as per RFC4570.
         *
         * @param [in] address_type: The address type.
         *
         * @return: Reference to the builder object.
         */
        Builder& set_address_type(AddressType address_type) { return set(m_instance->address_type, address_type); }
    };

protected:
    /**
     * @brief: Default constructor for SourceFilterAttribute.
     *
     * This constructor is protected and only accessible by the Builder class and its subclasses.
     */
    SourceFilterAttribute() = default;

private:
    NetworkFilterMode filter_mode = NetworkFilterMode::Inclusive;
    NetworkType network_type = NetworkType::_IN;
    AddressType address_type = AddressType::IP4;
    std::string destination_address;
    std::string source_list;

    friend class ISDP::IBuilder<SourceFilterAttribute, Builder>;
};
/**
 * @brief: SDP Media description builder.
 *
 * This class is responsible for constructing the common media description part of the SDP string.
 * It is based on the RFC4566 specification.
 *
 * The following common media description attributes are supported:
 * --------------------------------------------------------------------------------------------------------
 *     - Media Descriptions:
 *           m=<media> <port> <proto> <fmt>
 * --------------------------------------------------------------------------------------------------------
 *     - Connection Data:
 *           c=IN <addrtype> <connection-address>
 * --------------------------------------------------------------------------------------------------------
 */
class BaseMediaDescription : public ISDP
{
public:
    ~BaseMediaDescription() = default;
    operator json() const override final;
    /**
     * @brief: Returns the media description attributes.
     *
     * @return: A list of constructed media description JSON attributes.
     */
    virtual std::vector<json> get_media_description_attributes() const = 0;
    /**
     * @brief: Builder class for constructing BaseMediaDescription objects.
     *
     * @tparam ConcreteBuilder: The concrete builder object.
     * @tparam ConcreteMediaDescription: The concrete media description object.
     */
    template <typename ConcreteBuilder, typename ConcreteMediaDescription>
    class Builder : public ISDP::IBuilder<ConcreteMediaDescription, ConcreteBuilder>
    {
    public:
        /**
         * @brief: Constructor for mandatory parameters.
         *
         * @param [in] media_type: The media type.
         * @param [in] transport_port: The transport port.
         * @param [in] transport_protocol: The transport protocol.
         * @param [in] media_format_description: The media format description.
         * @param [in] connection_address: The connection address.
         */
        explicit Builder(
            MediaType media_type,
            size_t transport_port,
            TransportProtocol transport_protocol,
            const std::string& media_format_description,
            const std::string& connection_address)
            : ISDP::IBuilder<ConcreteMediaDescription, ConcreteBuilder>()
        {
            ISDP::IBuilder<ConcreteMediaDescription, ConcreteBuilder>::throw_if(
                media_type >= MediaType::Unknown || transport_port == 0 ||
                transport_protocol >= TransportProtocol::Unknown || media_format_description.empty() ||
                connection_address.empty()
            );

            this->m_instance->m_media_type = media_type;
            this->m_instance->m_transport_port = transport_port;
            this->m_instance->m_transport_protocol = transport_protocol;
            this->m_instance->m_media_format_description = media_format_description;
            this->m_instance->m_connection_address = connection_address;
        }

        // Setters for optional parameters:

        /**
         * @brief: Sets the address type.
         *
         * This corresponds to the <addrtype> field in "c=" line in SDP as per RFC4566.
         *
         * @param [in] address_type: The address type.
         *
         * @return: Reference to the builder object.
         */
        ConcreteBuilder& set_address_type(AddressType address_type)
        {
            this->m_instance->m_address_type = address_type;
            return static_cast<ConcreteBuilder&>(*this);
        }
        /**
         * @brief: Sets the connection TTL.
         *
         * This corresponds to the TTL in <connection-address> field in "c=" line in SDP as per RFC4566.
         *
         * @param [in] connection_ttl: The connection TTL.
         *
         * @return: Reference to the builder object.
         */
        ConcreteBuilder& set_connection_ttl(size_t connection_ttl)
        {
            this->m_instance->m_connection_ttl = connection_ttl;
            return static_cast<ConcreteBuilder&>(*this);
        }
    };

protected:
    /**
     * @brief: Default constructor for BaseMediaDescription.
     *
     * This constructor is protected and only accessible by the Builder class and its subclasses.
     */
    BaseMediaDescription() = default;
    /**
     * @brief: Returns the source filter attribute.
     *
     * @param [in] source_filter: The source filter attribute.
     *
     * @return: The constructed source filter JSON attribute.
     */
    json get_source_filter_attribute(const SourceFilterAttribute& source_filter) const { return source_filter; }
    /**
     * @brief: Returns the reference clock timestamp attribute.
     *
     * @param [in] timestamp_ref_clock: The timestamp reference clock.
     * @param [in] timestamp_ref_clock_ptp_grandmaster_clock_identity: The PTP grandmaster clock identity.
     * @param [in] timestamp_ref_clock_ptp_domain_number: The PTP domain number.
     * @param [in] timestamp_ref_clock_ptp_traceable: The traceable flag.
     * @param [in] timestamp_ref_clock_local_mac: The local MAC address.
     *
     * @return: The constructed reference clock timestamp JSON attribute.
     */
    json get_ref_clock_timestamp_attribute(
        TimestampRefClock timestamp_ref_clock,
        const std::string& timestamp_ref_clock_ptp_grandmaster_clock_identity,
        size_t timestamp_ref_clock_ptp_domain_number,
        bool timestamp_ref_clock_ptp_traceable,
        const std::string& timestamp_ref_clock_local_mac) const;
    /**
     * @brief: Returns the media clock attribute.
     *
     * @param [in] media_clock: The media clock.
     *
     * @return: The constructed media clock JSON attribute.
     */
    json get_media_clock_attribute(MediaClock media_clock) const;
    /**
     * @brief: Returns the media format specific attribute for the given formats.
     *
     * @param [in] formats: The media format specific attributes list.
     *
     * @return: The constructed media format specific JSON attribute.
     */
    json get_media_format_specific_attribute(const std::vector<MediaFormatAttribute>& formats) const;
    /**
     * @brief: Returns the RTP map attribute for the given formats.
     *
     * @param [in] formats: The RTP map attributes list.
     *
     * @return: The constructed RTP map JSON attribute.
     */
    json get_rtp_map_attribute(const std::vector<RTPMapAttribute>& formats) const;

    MediaType m_media_type = MediaType::Unknown;
    size_t m_transport_port = 0;
    TransportProtocol m_transport_protocol = TransportProtocol::Unknown;
    std::string m_media_format_description;
    AddressType m_address_type = AddressType::IP4;
    std::string m_connection_address;
    size_t m_connection_ttl = 64;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_SDP_SDP_COMMON_DESCRIPTIONS_H_ */
