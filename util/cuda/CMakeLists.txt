# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if (NOT CMAKE_CUDA_COMPILER)
    return()
endif()

project(rmax-api-apps-cuda)

#------------------------------------------------------------------------------

add_library(${PROJECT_NAME} STATIC)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
#set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(${PROJECT_NAME} PUBLIC GL_GLEXT_PROTOTYPES)
target_link_libraries(${PROJECT_NAME} PRIVATE rmax-apps-build)

target_link_libraries(${PROJECT_NAME} 
    PUBLIC
        CUDA::cuda_driver
        $<$<NOT:$<BOOL:${RMAX_TEGRA}>>:CUDA::nvml>
    PRIVATE 
        rmax-apps-util
)

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
