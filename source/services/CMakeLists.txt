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

project(rivermax-dev-kit-services)

add_library(${PROJECT_NAME} STATIC)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        Rivermax::Include
        rivermax-dev-kit-build
    PUBLIC
        rivermax-dev-kit-services-cpu
        rivermax-dev-kit-services-legacy-util
)
target_include_directories(${PROJECT_NAME} PUBLIC include)

set(RIVERMAX_DEV_KIT_SERVICE_COMPONENTS
    legacy_util
    error_handling
    settings
    cli
    cpu
    statistics
    utils
    media
    sdp
    buffer_wr
    memory_allocation
)
foreach(service_component ${RIVERMAX_DEV_KIT_SERVICE_COMPONENTS})
    add_subdirectory(${service_component})
endforeach()
