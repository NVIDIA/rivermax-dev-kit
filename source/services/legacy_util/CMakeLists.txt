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
#------------------------------------------------------------------------------
# Command-line options
#
#   These options are set ON via command line e.g.:
#       $ cmake -DRMAX_CUDA=ON
#
option(RMAX_CUDA     "Enables CUDA"   OFF)
option(RMAX_TEGRA    "Enables TEGRA"  OFF)
#------------------------------------------------------------------------------

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
        rivermax-dev-kit-services-cpu
    PUBLIC
        Rivermax::Include
        CLI::CLI
        $<$<BOOL:${RMAX_CUDA}>:rivermax-dev-kit-services-legacy-util-cuda>
)

target_sources(${PROJECT_NAME} 
    PUBLIC
        defs.h
        rational.h
        rt_threads.h
        $<$<BOOL:${RMAX_CUDA}>:${CMAKE_CURRENT_SOURCE_DIR}/gpu.h>
    PRIVATE
        rational.cpp
        rt_threads.cpp
        $<$<BOOL:${RMAX_CUDA}>:${CMAKE_CURRENT_SOURCE_DIR}/gpu.cpp>
)
