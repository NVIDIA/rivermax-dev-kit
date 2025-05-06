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

include(RivermaxDevKitBuild)

add_library(${PROJECT_NAME} STATIC)

target_sources(${PROJECT_NAME} PRIVATE facade.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        rivermax-dev-kit-build
    PUBLIC
        rivermax-dev-kit-core
        rivermax-dev-kit-services
        rivermax-dev-kit-io-node
        rivermax-dev-kit-apps
)

set(RIVERMAX_DEV_KIT_MODULES
    core
    services
    io_node
    apps
)
foreach(module ${RIVERMAX_DEV_KIT_MODULES})
    add_subdirectory(${module})
endforeach()
