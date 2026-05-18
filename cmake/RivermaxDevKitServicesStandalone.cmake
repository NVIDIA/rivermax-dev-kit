# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# ======================================================================
# RivermaxDevKitServicesStandalone
#
# Configures the RDK services pure tier as a standalone CMake project,
# isolated from the rest of the Rivermax dev-kit and from the compiled
# Rivermax SDK. This is a structural guard that ensures the pure tier
# does not silently regrow a dependency on Rivermax::Rivermax.
#
# Mechanics:
#   - Pre-creates ``Rivermax::Include`` as a benign INTERFACE target so
#     ``find_package(Rivermax)`` short-circuits and does not search for
#     an installed SDK.
#   - Pre-creates ``Rivermax::Rivermax`` as a *poison-pill* INTERFACE
#     target. Its INTERFACE_LINK_LIBRARIES references the unresolved
#     symbol ``Rivermax_binaries_not_allowed_for_services``, so any
#     pure-tier regression that links the SDK fails at link time with
#     an unmistakable error.
#   - Pulls in the standard dev-kit build interface (compile flags,
#     warning policy) and utility helpers used by the pure-tier
#     components.
#
# Usage::
#
#   if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
#       list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
#       include(RivermaxDevKitServicesStandalone)
#   endif()
#
# Callers are responsible for ensuring ``CMAKE_MODULE_PATH`` contains
# ``dev-kit/cmake/`` before including this file.
# ======================================================================

include_guard(GLOBAL)

project(rivermax-dev-kit-services-standalone
    DESCRIPTION "Isolation/standalone configuration for the RDK services pure tier."
    LANGUAGES CXX C
)

# Pre-create stub Rivermax targets so FindRivermax.cmake short-circuits
# before searching for an installed SDK.
add_library(rivermax-include-interface-standalone INTERFACE)
add_library(Rivermax::Include ALIAS rivermax-include-interface-standalone)

add_library(rivermax-library-interface-standalone INTERFACE)
target_link_libraries(rivermax-library-interface-standalone INTERFACE
    Rivermax_binaries_not_allowed_for_services)
add_library(Rivermax::Rivermax ALIAS rivermax-library-interface-standalone)

include(RivermaxDevKitBuild)
include(RivermaxDevKitUtilities)
