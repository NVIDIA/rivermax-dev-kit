# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(RivermaxDevKitAppSetup)

project(rivermax-dev-kit-apps)

# Create umbrella library to contain all RDK applications
add_library(${PROJECT_NAME} INTERFACE)

# Create base library to build an RDK application
add_library(rivermax-dev-kit-app-base STATIC)
target_sources(rivermax-dev-kit-app-base
    PRIVATE
        rmax_base_memory_strategy.cpp
        rmax_receiver_base.cpp
        rmax_base_app.cpp
)
target_include_directories(rivermax-dev-kit-app-base PUBLIC include)
target_link_libraries(rivermax-dev-kit-app-base
    PUBLIC
        rivermax-dev-kit-build
        rivermax-dev-kit-core
        rivermax-dev-kit-services
        rivermax-dev-kit-io-node
        rivermax-dev-kit-facade
)
target_link_libraries(${PROJECT_NAME} INTERFACE rivermax-dev-kit-app-base)

# Process all RDK applications
set(RIVERMAX_DEV_KIT_APP_COMPONENTS
    rmax_ipmx_receiver
    rmax_ipmx_sender
    rmax_latency
    rmax_ipo_receiver
    rmax_rtp_receiver
    rmax_media_probe
    rmax_xstream_media_sender
    rmax_xstream_generic_sender
    CACHE INTERNAL "Rivermax Dev Kit application components."
)
foreach(app_component ${RIVERMAX_DEV_KIT_APP_COMPONENTS})
    add_subdirectory(${app_component})
endforeach()
