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
    # Reconsider not to remove the following optimization
    foreach(lang_name C CXX)
        string(REPLACE "/Ob2" "" CMAKE_${lang_name}_FLAGS_RELEASE "${CMAKE_${lang_name}_FLAGS_RELEASE}")
    endforeach()
    set(RMAX_C_CXX_FLAGS
        /D_WINSOCK_DEPRECATED_NO_WARNINGS
        /DNOMINMAX
        /GS /GL /W4
        /Zc:wchar_t-
        /Zi /Gm-
        /Zc:inline /fp:fast
        /D_WIN64
        /D_AMD64_
        /DAMD64
        /DWIN32_LEAN_AND_MEAN=1
        /D_WIN32_WINNT=0x0A00
        /DWINVER=0x0A00
        /DWINNT=1
        /DNTDDI_VERSION=0xA00000A
        /D_STL100_
        /DUNICODE /D_UNICODE
        /D_ALLOW_RUNTIME_LIBRARY_MISMATCH
        /D_STATIC_CPPLIB
        /D_DISABLE_DEPRECATE_STATIC_CPPLIB
        /D_CRT_SECURE_NO_WARNINGS
        /D_HAS_ITERATOR_DEBUGGING=0
        /D_SECURE_SCL=0
        /D_SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS
        /D_CRTIMP_=
        /DNDEBUG
        /errorReport:prompt
        /WX
        /Zc:forScope
        /GR /Gz /MD /FC
        /EHsc /nologo
        /wd4324
        /wd4702
        /wd4100
        /wd4189
        /wd4459
        /wd4244
        /wd4245
        /wd4706
    )
    set(RMAX_CXX_FLAGS "")
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
    set(RMAX_CXX_FLAGS
        -Wno-overloaded-virtual
        -Woverloaded-virtual
        -Wnon-virtual-dtor
    )
endif()
target_compile_options(rivermax-dev-kit-build INTERFACE
    $<$<COMPILE_LANGUAGE:CXX,C>:${RMAX_C_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${RMAX_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-m64>
)
target_compile_definitions(rivermax-dev-kit-build INTERFACE
    ENABLE_DPCP
    CONFIG_MERSENNE_TWISTER
    $<$<BOOL:${RMAX_CUDA}>:CUDA_ENABLED>
    $<$<BOOL:${RMAX_TEGRA}>:TEGRA_ENABLED>
)
target_compile_features(rivermax-dev-kit-build INTERFACE $<$<BOOL:${RMAX_CUDA}>:cxx_std_11>)
find_package(Threads REQUIRED)
target_link_libraries(rivermax-dev-kit-build INTERFACE
    Threads::Threads
    $<$<BOOL:${RMAX_CUDA}>:CUDA::cuda_driver>
    $<$<BOOL:${RMAX_CUDA}>:CUDA::cudart>
    $<$<BOOL:${RMAX_CUDA}>:$<$<NOT:$<BOOL:${RMAX_TEGRA}>>:CUDA::nvml>>
)
set_target_properties(rivermax-dev-kit-build PROPERTIES INTERFACE_POSITION_INDEPENDENT_CODE ON)

if (CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets -D_ALLOW_RUNTIME_LIBRARY_MISMATCH")
endif()
