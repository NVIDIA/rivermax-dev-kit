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

if(RMAX_DEBUG_TOOLS)
    if(NOT CMAKE_PROPERTY_LIST)
        execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)

        # Convert command output into a CMake list
        string(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
        string(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    endif()
endif()

function(print_target_properties target)
    if(NOT RMAX_DEBUG_TOOLS)
        message(WARNING "Command line option `RMAX_DEBUG_TOOLS` shall be set to print properties.")
        return()
    endif()
    if(NOT TARGET ${target})
        message(STATUS "There is no target named '${target}'")
        return()
    endif()

    message("Listing properties: ${target}")

    foreach(property ${CMAKE_PROPERTY_LIST})
        string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" property ${property})

        # Fix https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
        if(property STREQUAL "LOCATION" OR property MATCHES "^LOCATION_" OR property MATCHES "_LOCATION$")
            continue()
        endif()

        get_property(was_set TARGET ${target} PROPERTY ${property} SET)
        if(was_set)
            get_target_property(value ${target} ${property})
            message("${target} ${property} = ${value}")
        endif()
    endforeach()
endfunction()

function(PrintLinkLibraries target)
    get_target_property(interf_libs ${target} INTERFACE_LINK_LIBRARIES)
    get_target_property(imported_libs ${target} IMPORTED_LINK_INTERFACE_LIBRARIES)
    get_target_property(l_libs ${target} LINK_LIBRARIES)
    get_target_property(li_libs ${target} LINK_INTERFACE_LIBRARIES)
    if (${ARGN})
        set(extra_text "\(${ARGN}\)")
    endif()
    message("Target ${target}${extra_text}, linked libraries:")
    message(STATUS ${interf_libs})
    message(STATUS ${imported_libs})
    message(STATUS ${l_libs})
    message(STATUS ${li_libs})
endfunction()