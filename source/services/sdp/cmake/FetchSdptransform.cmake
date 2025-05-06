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

include(FetchContent)

# Ensure MSVC static runtime is used, needed due to compatibility issues:
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

message("-- Fetching sdptransform...")

FetchContent_Declare(
    sdptransform
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    URL https://github.com/ibc/libsdptransform/archive/a6476c5e9411f37c5798ec72b00a53d19b9a3f95.zip
    # Update the CMakeLists.txt to fit our needs, this is temporary until we will have a fork of the project.
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/libsdptransform_CMakeLists.txt CMakeLists.txt
)
FetchContent_MakeAvailable(sdptransform)
