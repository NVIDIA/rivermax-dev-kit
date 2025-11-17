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

cmake_minimum_required(VERSION 3.19)

#------------------------------------------------------------------------------
# Command-line options
#
#   These options are set ON via command line e.g.:
#       $ cmake -DRMAX_CUDA=ON
#
option(RMAX_CUDA        "Enables CUDA"   OFF)
option(RMAX_TEGRA       "Enables TEGRA"  OFF)
option(RMAX_LINK_STATIC "Link against static Rivermax library" OFF)
#------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(rivermax-dev-kit
    VERSION 1.0.0
    DESCRIPTION "Rivermax development kit designed for Rivermax SDK."
    HOMEPAGE_URL https://github.com/NVIDIA/rivermax-dev-kit
    LANGUAGES CXX C
)
include(RivermaxDevKitBuild)

enable_testing()

add_subdirectory(source)
add_subdirectory(binaries)
add_subdirectory(utests)
add_subdirectory(examples)
add_subdirectory(docs)
