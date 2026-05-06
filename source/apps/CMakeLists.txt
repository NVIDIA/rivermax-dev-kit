# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
target_include_directories(${PROJECT_NAME} INTERFACE include)

# Create base library to build an RDK application
add_library(rivermax-dev-kit-apps-base STATIC)
target_sources(rivermax-dev-kit-apps-base
    PRIVATE
        base_memory_strategy.cpp
        receiver_base_app.cpp
        base_app.cpp
)
target_include_directories(rivermax-dev-kit-apps-base PUBLIC include)
target_link_libraries(rivermax-dev-kit-apps-base
    PUBLIC
        rivermax-dev-kit-build
        rivermax-dev-kit-core
        rivermax-dev-kit-services
        rivermax-dev-kit-io-node
        rivermax-dev-kit-facade
)
target_link_libraries(${PROJECT_NAME} INTERFACE rivermax-dev-kit-apps-base)

# Process all RDK applications
set(RIVERMAX_DEV_KIT_APP_COMPONENTS
    ipmx_receiver
    ipmx_sender
    latency
    ipo_receiver
    rtp_receiver
    media_probe
    # TODO: Enable media_receiver when the work is complete.
    # media_receiver
    media_sender
    generic_sender
    CACHE INTERNAL "Rivermax Dev Kit application components."
)
foreach(app_component ${RIVERMAX_DEV_KIT_APP_COMPONENTS})
    add_subdirectory(${app_component})
endforeach()
