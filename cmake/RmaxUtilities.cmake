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
#[=======================================================================[.rst:
RmaxUtilities
-------------

This is a module of utilities used by Rivermax CMAKE infrastructure
#]=======================================================================]

include_guard(GLOBAL)

#[=======================================================================[.rst:
.. command:: RmaxSetBuildType

  Set the default build type to a value ``default_type`` (passed to the
  compiler) selected from the list of build types supported by CMAKE.

  .. code-block:: cmake

    RmaxSetBuildType(<default_type>)
#]=======================================================================]
macro(RmaxSetBuildType default_type)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "${default_type}" CACHE STRING "Choose Release/Debug/RelWithDebInfo/..." FORCE)
    endif()
endmacro()

#[=======================================================================[.rst:
.. command:: RmaxDetermineVersion

  Extract from given header file ``version_file_path`` the Rivermax 
  version and export it in a standard format via ``out_version``

  .. code-block:: cmake

    RmaxDetermineVersion(<version_file_path> <out_version>)
#]=======================================================================]
function(RmaxDetermineVersion version_file_path out_version)
    file(STRINGS ${version_file_path} version_lines REGEX "^#define[ \t]+RMX_VERSION_(MAJOR|MINOR|PATCH)[ \t]")
    if (version_lines)
        foreach(_version_item "MAJOR" "MINOR" "PATCH")
            string(REGEX REPLACE ".*${_version_item}[ \t]+([0-9]+).*" "\\1" version_${_version_item} "${version_lines}")
        endforeach()
        set(${out_version} "${version_MAJOR}.${version_MINOR}.${version_PATCH}" PARENT_SCOPE)
    else()
        unset(${out_version} PARENT_SCOPE)
        message(WARNING "Failed to determine the version of Rivermax.")
    endif()
endfunction()

#[=======================================================================[.rst:
.. command:: RmaxListTargets

  List all targets defined within the directory tree specified 
  by ``root_directory`` and populate the list via ``output_var``.

  .. code-block:: cmake

    RmaxListTargets(<output_var> <root_directory>)
#]=======================================================================]
function(RmaxListTargets output_var root_directory)
    if (NOT root_directory)
        set(root_directory ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    get_directory_property(all_immediate_targets DIRECTORY ${root_directory} BUILDSYSTEM_TARGETS)
    set(tree_targets "")
    foreach(a_target ${all_immediate_targets})
        get_target_property(target_type ${a_target} TYPE)
        if ((target_type STREQUAL "EXECUTABLE") OR (target_type STREQUAL "SHARED_LIBRARY"))
            list(APPEND tree_targets ${a_target})
        endif()
    endforeach()

    get_directory_property(sub_directories DIRECTORY ${root_directory} SUBDIRECTORIES)
    foreach(sub_directory ${sub_directories})
        RmaxListTargets(sub_dir_targets ${sub_directory})
        list(APPEND tree_targets ${sub_dir_targets})
    endforeach()

    set(${output_var} ${tree_targets} PARENT_SCOPE)
endfunction()

include(CheckCXXCompilerFlag)
#[=======================================================================[.rst:
.. command:: RmaxAppendCompileFlags

  This function is analogous to AX_APPEND_COMPILE_FLAGS macro from 
  Autoconf Archive. Its purpose to add to given variable ``target_var``
  compilation flags passed to the function as part of ``ARGN``.

  .. code-block:: cmake

    RmaxAppendCompileFlags(<target_var> [flag #1] [flag #2] ...)
#]=======================================================================]
function(RmaxAppendCompileFlags target_var)
    foreach(compile_flag ${ARGN})
        STRING(REGEX REPLACE "[^A-Za-z0-9]" "_" VarName ${compile_flag})
        check_cxx_compiler_flag(${compile_flag} ${VarName})
        if (${VarName})
            list(APPEND ${target_var} ${compile_flag})
        endif()
    endforeach()
    set(${target_var} ${${target_var}} PARENT_SCOPE)
endfunction()

#[=======================================================================[.rst:
.. command:: rmax_find_static_library

  Find static library ``var_name``.

  .. code-block:: cmake

    RmaxListTargets(<output_var> <root_directory>)
#]=======================================================================]
function(rmax_find_static_library var_name var_option)
    if(WIN32)
        set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
    else()
        set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
    endif()
    find_library(${var_name} ${ARGV})
    if ("NO_CACHE" IN_LIST ${ARGN})
        if (${var_name})
            set(${var_name} ${${var_name}} PARENT_SCOPE)
        else()
            set(${var_name}-NOTFOUND ${${var_name}-NOTFOUND$} PARENT_SCOPE)
        endif()
    endif()
endfunction()
