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

#ifndef RDK_SERVICES_SDP_SDP_MANAGER_H_
#define RDK_SERVICES_SDP_SDP_MANAGER_H_

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "sdptransform.hpp"

#include "rdk/services/utils/enum_utils.h"
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
 * @brief: Session Description Protocol (SDP) manager.
 *
 * This class is responsible for managing the SDP string construction.
 */
class SDPManager : public ISDP
{
public:
    ~SDPManager() = default;
    /**
     * @brief: Converts the object to a string implicitly.
     *
     * @return: The object as a string.
     */
    operator std::string() const;
    /**
     * @brief: Converts the object to a string.
     *
     * @return: The object as a string.
     */
    std::string to_string() const { return operator std::string(); }
    /**
     * @brief: Converts the object to an output stream implicitly.
     *
     * @param [in] out: The output stream.
     * @param [in] sdp: The SDP object reference.
     *
     * @return: The object as a string stream.
     */
    friend std::ostream& operator<<(std::ostream& out, const SDPManager& sdp) { return out << std::string(sdp); }
    /**
     * @brief: Converts the object to an output stream implicitly.
     *
     * @param [in] out: The output stream.
     * @param [in] sdp: The SDP object pointer.
     *
     * @return: The object as a string stream.
     */
    friend std::ostream& operator<<(std::ostream& out, SDPManager* sdp) { return out << std::string(*sdp); }
    /**
     * @brief: Converts the object to an output stream implicitly.
     *
     * @param [in] out: The output stream.
     * @param [in] sdp: The SDP object std::unique_ptr.
     *
     * @return: The object as a string stream.
     */
    friend std::ostream& operator<<(std::ostream& out, const std::unique_ptr<SDPManager>& sdp) { return out << *sdp; }
    /**
     * @brief: Generates a Network Time Protocol (NTP) ID.
     *
     * This function generates a NTP ID using the NTP format, which recommended by RFC4566.
     *
     * @param [in] offset_seconds: The offset in seconds (default: 0).
     *
     * @return: The generated NTP ID.
     */
    static size_t generate_ntp_id(size_t offset_seconds = 0);
    /**
     * @brief: Builder class for constructing SDP objects.
     */
    class Builder : public ISDP::IBuilder<SDPManager, Builder>
    {
    public:
        /**
         * @brief: Constructor for mandatory parameters.
         *
         * @param [in] session_description: The session description.
         * @param [in] time_description: The time description.
         */
        explicit Builder(
            std::unique_ptr<SessionDescription> session_description, std::unique_ptr<TimeDescription> time_description)
            : ISDP::IBuilder<SDPManager, Builder>()
        {
            throw_if(!session_description || !time_description);

            m_instance->m_session_description = std::move(session_description);
            m_instance->m_time_description = std::move(time_description);
        }

        // Setters for optional parameters:

        /**
         * @brief: Adds a media description component to the SDP.
         *
         * @param [in] media_description: The media description.
         *
         * @return: Reference to the builder object.
         */
        Builder& add_media_description(std::unique_ptr<BaseMediaDescription> media_description)
        {
            m_instance->m_media_descriptions.push_back(std::move(media_description));
            return *this;
        }
    };

private:
    SDPManager() = default;
    operator json() const override;

    std::unique_ptr<SessionDescription> m_session_description;
    std::unique_ptr<TimeDescription> m_time_description;
    std::vector<std::unique_ptr<BaseMediaDescription>> m_media_descriptions;

    friend class ISDP::IBuilder<SDPManager, Builder>;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_SDP_SDP_MANAGER_H_ */
