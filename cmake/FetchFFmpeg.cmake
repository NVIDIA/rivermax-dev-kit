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

include(FetchContent)

list(APPEND CMAKE_PREFIX_PATH ${PROJECT_BINARY_DIR})
message("-- Detecting/fetching ffmpeg...")

set(RIVERMAX_FFMPEG_AUTOBUILD_TAG "autobuild-2026-01-31-12-57")
set(RIVERMAX_FFMPEG_BUILD_VERSION "N-122607-g50bcc96a75")

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows" AND ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64")
    set(RIVERMAX_FFMPEG_RELEASE "ffmpeg-${RIVERMAX_FFMPEG_BUILD_VERSION}-win64-lgpl-shared.zip")
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64" OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "amd64")
        set(RIVERMAX_FFMPEG_RELEASE "ffmpeg-${RIVERMAX_FFMPEG_BUILD_VERSION}-linux64-lgpl-shared.tar.xz")
    elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL aarch64)
        set(RIVERMAX_FFMPEG_RELEASE "ffmpeg-${RIVERMAX_FFMPEG_BUILD_VERSION}-linuxarm64-lgpl-shared.tar.xz")
    endif()
endif()

if(NOT RIVERMAX_FFMPEG_RELEASE)
    message(FATAL_ERROR "Platform ${CMAKE_SYSTEM_PROCESSOR} on system ${CMAKE_SYSTEM_NAME} is not supported!")
endif()

if (NOT "${PROJECT_BINARY_DIR}/.local" IN_LIST "${CMAKE_PREFIX_PATH}")
    list(APPEND CMAKE_PREFIX_PATH "${PROJECT_BINARY_DIR}/.local")
endif()

set(ffmpeg_source_url "https://github.com/BtbN/FFmpeg-Builds/releases/download/${RIVERMAX_FFMPEG_AUTOBUILD_TAG}/${RIVERMAX_FFMPEG_RELEASE}")
if(POLICY CMP0135)
    list(APPEND ffmpeg_declaration_extra DOWNLOAD_EXTRACT_TIMESTAMP ON)
endif()
FetchContent_Declare(FFmpeg URL "${ffmpeg_source_url}" SOURCE_DIR "${PROJECT_BINARY_DIR}/.local" ${ffmpeg_declaration_extra})
FetchContent_MakeAvailable(FFmpeg)
