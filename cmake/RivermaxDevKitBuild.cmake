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

include_guard(GLOBAL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(RMAX_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
endif()

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
elseif(RMAX_CUDA)
    message(FATAL_ERROR "Failed to find CUDA on this machine!")
elseif(RMAX_TEGRA)
    message(WARNING "Enabling of TEGRA requires CUDA!")
endif()

add_library(rivermax-dev-kit-build INTERFACE)

if (MSVC)
    set(RDK_C_CXX_FLAGS
        /W4
        /WX
        /fp:fast
        /GL
    )
    target_compile_options(rivermax-dev-kit-build INTERFACE $<$<COMPILE_LANGUAGE:C,CXX>:${RDK_C_CXX_FLAGS}>)
    target_link_options(rivermax-dev-kit-build INTERFACE /LTCG)

    set_target_properties(rivermax-dev-kit-build PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

    target_compile_definitions(rivermax-dev-kit-build INTERFACE
        WIN32_LEAN_AND_MEAN=1
        NOMINMAX
        _WINSOCK_DEPRECATED_NO_WARNINGS
        _CRT_SECURE_NO_WARNINGS
        UNICODE
        _UNICODE
    )

    set(RDK_MSVC_DEBUG_MODE_WARNINGS
        _SCL_SECURE_NO_WARNINGS
        _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
        _SILENCE_ALL_MS_EXT_DEPRECATION_WARNINGS
    )
    target_compile_definitions(rivermax-dev-kit-build INTERFACE $<$<CONFIG:Debug>:${RDK_MSVC_DEBUG_MODE_WARNINGS}>)

    # Define the Windows version and the minimum required Windows version as Windows 10 20H2 (Oct 2020),
    # to allow using features like: High-Performance Networking, NUMA Improvements, Large Pages, Modern Synchronization.
    target_compile_definitions(rivermax-dev-kit-build INTERFACE
        WINVER=0x0A00
        _WIN32_WINNT=0x0A00
        NTDDI_VERSION=0x0A00000A
    )

    set(RDK_MSVC_DISABLED_WARNINGS
        /wd4189 # unused loop iterators
        /wd4244 # type conversion with possible loss of data
        /wd4100 # unreferenced formal parameter
    )
    target_compile_options(rivermax-dev-kit-build INTERFACE $<$<COMPILE_LANGUAGE:C,CXX>:${RDK_MSVC_DISABLED_WARNINGS}>)

else()
    set(RMAX_C_CXX_FLAGS
        -g
        -Wall
        -Wextra
        -Werror
        -Wno-unknown-pragmas
        -ffunction-sections
        -fdata-sections
        -pipe
        -Wmissing-include-dirs
        -Wunused-local-typedefs
        -Wvla
        -Wno-switch
        -Wno-restrict
        -Wno-sign-compare
        -Wno-unused-variable
        -Wno-unused-function
        -Wno-unused-parameter
        -Wno-stringop-truncation
        -Wno-unused-but-set-variable
    )
    set(RMAX_CXX_ONLY_FLAGS
        -Wno-overloaded-virtual
        -Woverloaded-virtual
        -Wnon-virtual-dtor
    )
    target_compile_options(rivermax-dev-kit-build INTERFACE
        $<$<COMPILE_LANGUAGE:C,CXX>:${RMAX_C_CXX_FLAGS}>
        $<$<COMPILE_LANGUAGE:CXX>:${RMAX_CXX_ONLY_FLAGS}>
    )
endif()

target_compile_options(rivermax-dev-kit-build INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-m64>
)

target_compile_definitions(rivermax-dev-kit-build INTERFACE
    $<$<BOOL:${RMAX_CUDA}>:CUDA_ENABLED>
    $<$<BOOL:${RMAX_TEGRA}>:TEGRA_ENABLED>
)

target_compile_features(rivermax-dev-kit-build INTERFACE $<$<BOOL:${RMAX_CUDA}>:cxx_std_11>)

find_package(Rivermax REQUIRED)
find_package(Threads REQUIRED)

target_link_libraries(rivermax-dev-kit-build INTERFACE
    Rivermax::Include
    Threads::Threads
    $<$<BOOL:${RMAX_CUDA}>:CUDA::cuda_driver>
    $<$<BOOL:${RMAX_CUDA}>:CUDA::cudart>
    $<$<BOOL:${RMAX_CUDA}>:$<$<NOT:$<BOOL:${RMAX_TEGRA}>>:CUDA::nvml>>
)

set_target_properties(rivermax-dev-kit-build PROPERTIES INTERFACE_POSITION_INDEPENDENT_CODE ON)

if (CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets -D_ALLOW_RUNTIME_LIBRARY_MISMATCH")
endif()
