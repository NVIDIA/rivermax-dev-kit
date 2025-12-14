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

project(rivermax-dev-kit-services-legacy-util)

add_library(${PROJECT_NAME} STATIC)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(CLI)
add_subdirectory(os)
if (CMAKE_CUDA_COMPILER)
    add_subdirectory(cuda)
endif()

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        rivermax-dev-kit-build
        rivermax-dev-kit-services-cpu-affinity
    PUBLIC
        Rivermax::Include
        CLI::CLI
        rivermax-dev-kit-services-utils
        $<$<BOOL:${RMAX_CUDA}>:rivermax-dev-kit-services-legacy-util-cuda>
)

target_sources(${PROJECT_NAME}
    PUBLIC
        defs.h
        rt_threads.h
        gpu.h
    PRIVATE
        rt_threads.cpp
        gpu.cpp
)
