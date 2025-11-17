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
.. command:: create_rdk_app_objects

    This function creates an RDK application library of compiled objects that
    can be linked with any other software component.

    It accepts the following arguments:

    ``NAME``
      The name of the library.
    ``SOURCES``
      (Optional) Source files to include in the library.

    Example usage:

    .. code-block:: cmake

      create_rdk_app_objects(
          NAME MyApp_obj
          SOURCES MyApp.cpp extra_source1.cpp extra_source2.cpp
      )

    This will create a object library named MyApp_obj with sources 
    MyApp.cpp and extra_source1.cpp, extra_source2.cpp.
#]=======================================================================]
function(create_rdk_app_objects)
    set(options)
    set(one_value_args NAME)
    set(multi_value_args SOURCES)
    cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT ARGS_NAME)
        message(FATAL_ERROR "NAME argument is required for create_rdk_app_objects")
    endif()

    # Create the library target
    add_library(${ARGS_NAME} OBJECT)
    target_sources(${ARGS_NAME} PRIVATE ${ARGS_SOURCES})
    target_include_directories(${ARGS_NAME} PUBLIC include)
    target_link_libraries(${ARGS_NAME} PRIVATE rivermax-dev-kit-apps-base)

    # Add to the umbrella library of all the applications
    target_link_libraries(rivermax-dev-kit-apps INTERFACE ${ARGS_NAME})
endfunction()

#[=======================================================================[.rst:
.. command:: create_rdk_app_executable

  This function creates an RDK application executable and links it with its library. 
  It accepts the following arguments:

  ``NAME``
    The name of the executable.
  ``SOURCES``
    (Optional) Source files to include in the executable.
  ``LIBRARIES``
    (Optional) Libraries to link with the executable.

  Example usage:

  .. code-block:: cmake

    create_rdk_app_executable(
        NAME MyApp
        SOURCES MyApp_main.cpp
        LIBRARIES MyApp_lib
    )

  This will create an executable named MyApp with the main source 
  MyApp_main.cpp, linked with the library MyApp_lib.
#]=======================================================================]
function(create_rdk_app_executable)
    set(options)
    set(one_value_args NAME)
    set(multi_value_args SOURCES LIBRARIES)
    cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT ARGS_NAME)
        message(FATAL_ERROR "NAME argument is required for create_rdk_app_executable")
    endif()

    # Create the executable target
    add_executable(${ARGS_NAME})
    target_sources(${ARGS_NAME} PRIVATE ${ARGS_SOURCES})
    target_link_libraries(${ARGS_NAME}
        PRIVATE
            rivermax-dev-kit-apps-base
            ${ARGS_LIBRARIES}
    )
endfunction()
