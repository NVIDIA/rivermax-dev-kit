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

project(rivermax-dev-kit-services-legacy-util-cuda)

add_library(${PROJECT_NAME} STATIC)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)

target_sources(${PROJECT_NAME}
    PRIVATE
        checksum_kernel.cu
        csc_kernel.cu
    PUBLIC
        checksum_kernel.h
        csc_kernel.h
        csc_server.h
        csc_utils.h
)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(${PROJECT_NAME} PUBLIC GL_GLEXT_PROTOTYPES)
target_link_libraries(${PROJECT_NAME} PRIVATE rivermax-dev-kit-build)

include(FetchContent)

FetchContent_Declare(
    cuda_samples
    GIT_REPOSITORY https://github.com/NVIDIA/cuda-samples.git
    GIT_TAG        v13.1
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  _unused
)

FetchContent_MakeAvailable(cuda_samples)

if(UNIX)
    file(REMOVE_RECURSE "${cuda_samples_SOURCE_DIR}/Common/GL")
endif()

add_library(rivermax-dev-kit-cuda-helpers INTERFACE)
target_include_directories(rivermax-dev-kit-cuda-helpers INTERFACE
    ${cuda_samples_SOURCE_DIR}/Common
)

target_link_libraries(${PROJECT_NAME} PUBLIC rivermax-dev-kit-cuda-helpers)

if(MSVC)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13)
        file(GLOB cuda_sources "*.cu")
        set_source_files_properties(${cuda_sources} PROPERTIES
            VS_SETTINGS "CudaRuntime=Hybrid")
        target_link_options(${PROJECT_NAME} INTERFACE /NODEFAULTLIB:LIBCMT)
    endif()
    set_target_properties(${PROJECT_NAME} PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
        # Disable LTO/IPO for CUDA - nvcc doesn't fully support MSVC's /GL and /LTCG flags
        INTERPROCEDURAL_OPTIMIZATION OFF
    )
endif()
