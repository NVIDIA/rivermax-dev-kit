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

cmake_minimum_required(VERSION 3.19)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
    include(RivermaxDevKitServicesStandalone)
    set(RIVERMAX_DEV_KIT_SERVICES_STANDALONE ON)
endif()

project(rivermax-dev-kit-services)

# Define the umbrella INTERFACE target up-front so each component can self-register
# into it via target_link_libraries(rivermax-dev-kit-services INTERFACE ${PROJECT_NAME}).
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE include)

foreach(_component
    error_handling
    utils
    media
    sdp
    protocol
    settings
    cli
    memory_allocation
    legacy_util
    cpu
    ulp_packet
    ulp_packet_buffer
    statistics
)
    add_subdirectory(${_component})
endforeach()
