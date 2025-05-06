# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(rmax-apps-util)

add_subdirectory(CLI)

add_library(rmax-apps-util-reduced STATIC)
target_sources(rmax-apps-util-reduced PRIVATE
    rational.cpp
    rational.h
    memory_allocator.cpp
    memory_allocator.h
    rt_threads.cpp
    rt_threads.h
    defs.h
    fifo.h
    $<$<BOOL:${RMAX_CUDA}>:${CMAKE_CURRENT_SOURCE_DIR}/gpu.cpp>
    $<$<BOOL:${RMAX_CUDA}>:${CMAKE_CURRENT_SOURCE_DIR}/gpu.h>
)
target_link_libraries(rmax-apps-util-reduced PUBLIC
    CLI::CLI
    rivermax_libs_processor
    rmax-apps-build
    $<$<BOOL:${RMAX_CUDA}>:rmax-api-apps-cuda>
)

add_library(${PROJECT_NAME} STATIC)
target_sources(${PROJECT_NAME} PRIVATE
    rational.cpp
    rational.h
    memory_allocator.cpp
    memory_allocator.h  
    rt_threads.cpp
    rt_threads.h
    defs.h
    fifo.h
    checksum_header.h
    gpu.h
    $<$<BOOL:${RMAX_CUDA}>:${CMAKE_CURRENT_SOURCE_DIR}/gpu.cpp>
)
target_compile_definitions(${PROJECT_NAME} PUBLIC 
    $<$<BOOL:${RMAX_CUDA}>:CUDA_ENABLED>
    $<$<BOOL:${RMAX_TEGRA}>:TEGRA_ENABLED>
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    CLI::CLI
    rivermax_libs_processor
    rmax-apps-build
    $<$<BOOL:${RMAX_CUDA}>:rmax-api-apps-cuda>
)

add_subdirectory(os)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
    add_subdirectory(cuda)
endif()
