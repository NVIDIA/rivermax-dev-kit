# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake) # standalone mode
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
endif()
include(RmaxUtilities)

project(rmax-ral-apps
    VERSION "0.1.0"  
    DESCRIPTION "Rivermax application library based on Rivermax SDK ${Rivermax_VERSION}" 
    HOMEPAGE_URL https://developer.nvidia.com/networking/rivermax
    LANGUAGES CXX C
)

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/util) # standalone mode
    include(RmaxAppsBuild)
    add_subdirectory(util)
endif()
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/libs) # standalone mode
    add_subdirectory(libs)
    target_link_libraries(rivermax_libs_config INTERFACE rmax-apps-build)    
endif()

add_library(rmax-ral-build INTERFACE)
target_sources(rmax-ral-build INTERFACE api/rmax_apps_lib_api.h)
target_include_directories(rmax-ral-build INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(rmax-ral-build INTERFACE rmax-apps-util-reduced rmax-apps-build)
set_target_properties(rmax-ral-build PROPERTIES INTERFACE_POSITION_INDEPENDENT_CODE ON)

foreach(lib_dir "lib" "io_node" "apps")
    add_subdirectory(${lib_dir})
endforeach()

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/docs/CMakeLists.txt)
    add_subdirectory(docs)
endif()

