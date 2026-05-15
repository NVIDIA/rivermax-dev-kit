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

project(rivermax-dev-kit-services-ulp-packet)

add_library(${PROJECT_NAME} STATIC)

target_include_directories(${PROJECT_NAME} PUBLIC include)

set(RIVERMAX_DEV_KIT_ULP_PACKET_COMPONENTS
    readers
    writers
)

foreach(packet_component ${RIVERMAX_DEV_KIT_ULP_PACKET_COMPONENTS})
    add_subdirectory(${packet_component})
endforeach()

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        rivermax-dev-kit-build
    PUBLIC
        rivermax-dev-kit-services-error-handling
        rivermax-dev-kit-services-media
        rivermax-dev-kit-services-memory-allocation
        rivermax-dev-kit-services-utils
)

target_link_libraries(rivermax-dev-kit-services INTERFACE ${PROJECT_NAME})
