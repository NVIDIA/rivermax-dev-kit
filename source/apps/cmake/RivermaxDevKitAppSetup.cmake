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

function(setup_rivermax_dev_kit_app)
    # This function sets up a Rivermax Dev Kit application by creating a static library
    # and an executable. It accepts the following arguments:
    #
    # APP_NAME: The name of the application. This will be used to name the library
    #           and executable.
    # EXTRA_LIB_SOURCES: (Optional) Additional source files to include in the library.
    #
    # Example usage:
    # setup_rivermax_dev_kit_app(
    #     APP_NAME MyApp
    #     EXTRA_LIB_SOURCES extra_source1.cpp extra_source2.cpp
    # )
    #
    # This will create:
    # - A static library named MyApp_lib with sources MyApp.cpp and extra_source1.cpp, extra_source2.cpp.
    # - An executable named MyApp with the main source MyApp_main.cpp.

    set(options)
    set(one_value_args APP_NAME)
    set(multi_value_args EXTRA_LIB_SOURCES)
    cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    set(APP_LIB_NAME ${ARGS_APP_NAME}_lib)
    set(APP_LIB_SOURCES ${ARGS_APP_NAME}.cpp)
    set(APP_MAIN_SOURCE ${ARGS_APP_NAME}_main.cpp)

    add_library(${APP_LIB_NAME} STATIC)

    target_sources(${APP_LIB_NAME} PRIVATE ${APP_LIB_SOURCES} ${ARGS_EXTRA_LIB_SOURCES})
    target_include_directories(${APP_LIB_NAME} PUBLIC include)
    target_link_libraries(${APP_LIB_NAME}
        PRIVATE
            rivermax-dev-kit-build
        PUBLIC
            rivermax-dev-kit-apps
    )
    target_link_libraries(rivermax-dev-kit-apps PUBLIC ${APP_LIB_NAME})

    add_executable(${ARGS_APP_NAME} ${APP_MAIN_SOURCE})

    target_include_directories(${ARGS_APP_NAME} PUBLIC include)
    target_link_libraries(${ARGS_APP_NAME}
        PRIVATE
            rivermax-dev-kit-build
        PUBLIC
            ${APP_LIB_NAME}
    )
endfunction()
