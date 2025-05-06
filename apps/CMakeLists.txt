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

project(rmax-base-app LANGUAGES CXX)

add_library(${PROJECT_NAME} STATIC)
target_link_libraries(${PROJECT_NAME} 
    PUBLIC 
        rmax-ral-build 
        rmax-ral-lib 
        rmax-ral-io-node
)
target_sources(${PROJECT_NAME}
    PRIVATE
        rmax_base_app.cpp
    PUBLIC
        rmax_base_app.h
)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR})

set(RMAX_XTREAM_APPS 
    "rmax_ipo_receiver"
    "rmax_xstream_generic_sender"
    "rmax_xstream_media_sender"
    "rmax_latency"
    CACHE STRING "Rivermax eXtream applications (based on Rivermax Application Library)."
)
foreach(lib_dir ${RMAX_XTREAM_APPS})
    add_subdirectory(${lib_dir})
endforeach()
