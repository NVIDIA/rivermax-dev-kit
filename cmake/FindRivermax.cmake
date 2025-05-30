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

#[=======================================================================[.rst:
FindRivermax
------------

Finds the Rivermax library.

Input Options
^^^^^^^^^^^^^^

``LINK_STATIC``
  [Optional] If set to TRUE, the static version of Rivermax will be found.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``Rivermax::Rivermax``
  The Rivermax library

``Rivermax::Include``
  The Rivermax include directory

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Rivermax_FOUND``
  True if the system has Rivermax.
``Rivermax_VERSION``
  The version of Rivermax found.

#]=======================================================================]

set(INCLUDE_PATH_SUFFIXES mellanox Rivermax/include mellanox/Rivermax/include)
set(LIBRARY_PATH_SUFFIXES Rivermax/lib mellanox/Rivermax/lib)

include(FindPackageHandleStandardArgs)
include(RivermaxDevKitUtilities)

# Check if Rivermax target already exists, and exit if it does
if (TARGET Rivermax::Rivermax)
    message(STATUS "Using locally compiled Rivermax version ${Rivermax_VERSION}.")
    return()
endif()

message(WARNING "Looking for Rivermax library.")

# Find Rivermax library
find_library(Rivermax_LIBRARY NAMES rivermax PATH_SUFFIXES ${LIBRARY_PATH_SUFFIXES})
find_path(Rivermax_INCLUDE_DIR rivermax_defs.h PATH_SUFFIXES ${INCLUDE_PATH_SUFFIXES})
mark_as_advanced(Rivermax_LIBRARY Rivermax_INCLUDE_DIR)
if (Rivermax_INCLUDE_DIR AND Rivermax_LIBRARY)
    RmaxDetermineVersion(${Rivermax_INCLUDE_DIR}/rivermax_defs.h Rivermax_VERSION)
    set(Rivermax_VERSION ${Rivermax_VERSION} CACHE INTERNAL "")

    # if user requested static library via LINK_STATIC option
    if (Rivermax_FIND_LINK_STATIC) 
        if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
            set(Rivermax_STATIC_NAME rivermax_static.lib)
        else()
            set(Rivermax_STATIC_NAME librivermax.a)
        endif()
        get_filename_component(Rivermax_LIBRARY_PATH ${Rivermax_LIBRARY} DIRECTORY)
        find_library(Rivermax_LIBRARY NAMES ${Rivermax_STATIC_NAME} PATHS ${Rivermax_LIBRARY_PATH} NO_DEFAULT_PATH)
    endif()
endif()

# On Windows, find WindOF2
if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
    list(APPEND Rivermax_FIND_COMPONENTS WindOF2)
    list(APPEND CMAKE_PREFIX_PATH $ENV{ProgramW6432}\\Mellanox)
    find_file(WinOF2_BUILD_FILE NAMES build_id.txt PATH_SUFFIXES MLNX_WinOF2)
    mark_as_advanced(WinOF2_BUILD_FILE)
    if (WinOF2_BUILD_FILE)
        set(Rivermax_WindOF2_FOUND TRUE)
        file(STRINGS "${WinOF2_BUILD_FILE}" Rivermax_WinOF2_VERSION REGEX "^Version:[ \t]*[0-9\.]+" )
        string(REGEX REPLACE "^Version:[ \t]*([0-9\.]+)" "\\1" Rivermax_WinOF2_VERSION "${Rivermax_WinOF2_VERSION}")
        message(STATUS "WindOF2 version ${Rivermax_WinOF2_VERSION} found!")
        mark_as_advanced(Rivermax_WinOF2_VERSION)
    else()
        unset(Rivermax_WindOF2_FOUND)
    endif()

# On Linux, find DPCP
elseif ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    list(APPEND Rivermax_FIND_COMPONENTS DPCP)
    find_library(DPCP_LIBRARY NAMES dpcp)
    mark_as_advanced(DPCP_LIBRARY)
    if (DPCP_LIBRARY)
        set(Rivermax_DPCP_FOUND TRUE)
        file(GLOB filename_with_version_number "${DPCP_LIBRARY}.*")
        string(REGEX REPLACE ".*\.([0-9]+\.[0-9]+\.[0-9]+)" "\\1" Rivermax_DPCP_VERSION "${filename_with_version_number}")
        message(STATUS "DPCP version ${Rivermax_DPCP_VERSION} found!")
        mark_as_advanced(Rivermax_DPCP_VERSION)
    else()
        unset(Rivermax_DPCP_FOUND)
    endif()
else()
    message(FATAL_ERROR "OS '${CMAKE_SYSTEM_NAME}' is currently not supported by Rivermax.")
endif()
find_package_handle_standard_args(Rivermax
  REQUIRED_VARS Rivermax_LIBRARY Rivermax_INCLUDE_DIR
  VERSION_VAR Rivermax_VERSION
  HANDLE_COMPONENTS
)

# Create imported targets
if(Rivermax_FOUND AND NOT TARGET Rivermax::Rivermax)
    add_library(Rivermax::Rivermax UNKNOWN IMPORTED)
    set_target_properties(Rivermax::Rivermax PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Rivermax_INCLUDE_DIR}"
        IMPORTED_LOCATION "${Rivermax_LIBRARY}"
        VERSION "${Rivermax_VERSION}"
    )
endif()

# Create include target
if(Rivermax_FOUND AND NOT TARGET Rivermax::Include)
    add_library(rivermax_include INTERFACE)
    target_include_directories(rivermax_include INTERFACE "${Rivermax_INCLUDE_DIR}")
    add_library(Rivermax::Include ALIAS rivermax_include)
endif()
