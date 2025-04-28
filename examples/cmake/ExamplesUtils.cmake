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
.. command:: add_example_executables

    Add example executables to the build system.

    This function is designed to simplify the process of adding example applications to the build system.
    It generates a binary for each example.

    Parameters:
        DIRECTORY (string, optional): The directory where the example source files are located.
                                      Defaults to the current source directory if not provided.
        EXAMPLES (list, optional): A list of example names.
                                   Defaults to all .cpp files in the directory if not provided.
        TARGETS (list): A list of target libraries to link with each example.

    .. code-block:: cmake

    add_example_executables(
        DIRECTORY path/to/examples (optional)
        EXAMPLES example1 example2 example3 (optional)
        TARGETS target1 target2
    )
#]=======================================================================]
function(add_example_executables)
    cmake_parse_arguments(ARG "" "DIRECTORY" "EXAMPLES;TARGETS" ${ARGN})

    if (NOT ARG_DIRECTORY)
        set(ARG_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    if(NOT ARG_EXAMPLES)
        file(GLOB ARG_EXAMPLES RELATIVE ${ARG_DIRECTORY} ${ARG_DIRECTORY}/*.cpp)
        string(REPLACE ".cpp" "" ARG_EXAMPLES "${ARG_EXAMPLES}")
    endif()

    foreach(example ${ARG_EXAMPLES})
        add_executable(${example} ${ARG_DIRECTORY}/${example}.cpp)

        target_link_libraries(${example} PRIVATE rivermax-dev-kit-examples-base ${ARG_TARGETS})
        target_include_directories(${example} PRIVATE include)

        set_property(GLOBAL APPEND PROPERTY RIVERMAX_DEV_KIT_EXAMPLE_EXECUTABLES ${example})
    endforeach()
endfunction()

#[=======================================================================[.rst:
.. command:: add_subdirectories

    Add subdirectories to the build system.

    This function is designed to simplify the process of adding subdirectories to the build system.
    It automatically adds all valid subdirectories in the specified directory.

    Parameters:
        DIRECTORY (string, optional): The directory where the subdirectories are located.
                                      Defaults to the current source directory if not provided.
        SUB_DIRECTORIES (list, optional): A list of subdirectory names.
                                          Defaults to all subdirectories in the directory if not provided.

    .. code-block:: cmake

    add_subdirectories(
        DIRECTORY path/to/subdirectories (optional)
        SUB_DIRECTORIES subdir1 subdir2 (optional)
    )
#]=======================================================================]
function(add_subdirectories)
    cmake_parse_arguments(ARG "" "DIRECTORY" "SUB_DIRECTORIES" ${ARGN})

    if (NOT ARG_DIRECTORY)
        set(ARG_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    if (NOT ARG_SUB_DIRECTORIES)
        file(GLOB ARG_SUB_DIRECTORIES RELATIVE ${ARG_DIRECTORY} ${ARG_DIRECTORY}/*)
        list(FILTER ARG_SUB_DIRECTORIES EXCLUDE REGEX "^(cmake|include)$")
    endif()

    foreach(subdir ${ARG_SUB_DIRECTORIES})
        if (IS_DIRECTORY ${ARG_DIRECTORY}/${subdir})
            add_subdirectory(${ARG_DIRECTORY}/${subdir})
        endif()
    endforeach()
endfunction()
