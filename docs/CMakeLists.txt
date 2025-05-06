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

# Set rivermax-dev-kit project/Doxygen variables:
set(RIVERMAX_DEV_KIT_DOXYFILE_PROJECT_NUMBER ${PROJECT_VERSION})
set(RIVERMAX_DEV_KIT_DOXYFILE_PROJECT_BRIEF ${PROJECT_DESCRIPTION})
set(RIVERMAX_DEV_KIT_PROJECT_ROOT_DIR ${PROJECT_SOURCE_DIR})

project(rivermax_dev_kit_docs)

# Verify Rivermax is available:
if (NOT TARGET Rivermax::Rivermax)
    message(WARNING "No Rivermax library was determined to generate the docs.")
    return()
endif()

# Verify Doxygen is available:
find_package(Doxygen COMPONENTS dot)
if (NOT TARGET Doxygen::doxygen)
    message(STATUS "No Doxygen tool was determined to generate the docs.")
    return()
endif()

# Set project variables:
set(RIVERMAX_DEV_KIT_PROJECT_SOURCE_DIR ${RIVERMAX_DEV_KIT_PROJECT_ROOT_DIR}/source)
set(RIVERMAX_DEV_KIT_PROJECT_CONTRIBUTING ${RIVERMAX_DEV_KIT_PROJECT_ROOT_DIR}/CONTRIBUTING.md)
set(RIVERMAX_DEV_KIT_PROJECT_LICENSE ${RIVERMAX_DEV_KIT_PROJECT_ROOT_DIR}/LICENSE.md)

# Set Doxygen project variables:
set(RIVERMAX_DEV_KIT_DOXYFILE_PROJECT_NAME "Rivermax Dev Kit")
set(RIVERMAX_DEV_KIT_DOXYFILE_PROJECT_README ${RIVERMAX_DEV_KIT_PROJECT_ROOT_DIR}/README.md)
set(RIVERMAX_DEV_KIT_DOXYFILE_PROJECT_IMAGES_PATH ${RIVERMAX_DEV_KIT_PROJECT_ROOT_DIR}/docs/images)
string(REPLACE ";" " " RIVERMAX_DEV_KIT_DOXYFILE_IMAGE_PATHS "${RIVERMAX_DEV_KIT_DOXYFILE_PROJECT_IMAGES_PATH}")
set(RIVERMAX_DEV_KIT_DOXYFILE_PROJECT_LOGO ${RIVERMAX_DEV_KIT_DOXYFILE_PROJECT_IMAGES_PATH}/logo_small.png)
set(RIVERMAX_DEV_KIT_DOXYFILE_EXCLUDE ${RIVERMAX_DEV_KIT_PROJECT_SOURCE_DIR}/services/legacy_util)

# Set project variable for the SW Stack image (used for README.md installation):
set(RIVERMAX_DEV_KIT_SW_STACK_IMAGE_PATH ${RIVERMAX_DEV_KIT_DOXYFILE_PROJECT_IMAGES_PATH}/sw_stack.png
    CACHE FILEPATH "${RIVERMAX_DEV_KIT_DOXYFILE_PROJECT_NAME} SW stack image"
)

# Set the input files for Doxygen:
set(RIVERMAX_INCLUDE_FILES
    rmx_stats_defs.h
    rmx_stats_api.h
    rivermax_defs.h
    rivermax_api.h
)
get_target_property(RIVERMAX_INTERFACE_INCLUDE Rivermax::Include INTERFACE_INCLUDE_DIRECTORIES)
list(TRANSFORM RIVERMAX_INCLUDE_FILES PREPEND ${RIVERMAX_INTERFACE_INCLUDE}/)

# Set the input list for Doxygen:
list(APPEND RIVERMAX_DEV_KIT_DOXYFILE_INPUT_LIST ${RIVERMAX_INCLUDE_FILES})
list(APPEND RIVERMAX_DEV_KIT_DOXYFILE_INPUT_LIST ${RIVERMAX_DEV_KIT_PROJECT_SOURCE_DIR})
list(APPEND RIVERMAX_DEV_KIT_DOXYFILE_INPUT_LIST ${RIVERMAX_DEV_KIT_DOXYFILE_PROJECT_README})
list(APPEND RIVERMAX_DEV_KIT_DOXYFILE_INPUT_LIST ${RIVERMAX_DEV_KIT_PROJECT_CONTRIBUTING})
list(APPEND RIVERMAX_DEV_KIT_DOXYFILE_INPUT_LIST ${RIVERMAX_DEV_KIT_PROJECT_LICENSE})
string(REPLACE ";" " " RIVERMAX_DEV_KIT_DOXYFILE_INPUT "${RIVERMAX_DEV_KIT_DOXYFILE_INPUT_LIST}")

# Set Doxygen input and output properties:
set(RIVERMAX_DEV_KIT_DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(RIVERMAX_DEV_KIT_DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
set(RIVERMAX_DEV_KIT_DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
set(RIVERMAX_DEV_KIT_DOXYGEN_HTML_DIR ${RIVERMAX_DEV_KIT_DOXYGEN_OUTPUT_DIR}/html)
set(RIVERMAX_DEV_KIT_DOXYGEN_INDEX_FILE ${RIVERMAX_DEV_KIT_DOXYGEN_HTML_DIR}/index.html)
set(RIVERMAX_DEV_KIT_DOXYFILE_WARN_LOGFILE ${RIVERMAX_DEV_KIT_DOXYGEN_OUTPUT_DIR}/doxygen_warnings.log)

# Ensure the output directories for Doxygen:
file(MAKE_DIRECTORY ${RIVERMAX_DEV_KIT_DOXYGEN_OUTPUT_DIR})

# Substitute variables inside the Doxyfile template with the actual values:
configure_file(${RIVERMAX_DEV_KIT_DOXYFILE_IN} ${RIVERMAX_DEV_KIT_DOXYFILE_OUT} @ONLY)

# Custom command to invoke Doxygen:
add_custom_command(
    OUTPUT ${RIVERMAX_DEV_KIT_DOXYGEN_INDEX_FILE}
    COMMENT "Generating HTML documentation..."
    DEPENDS ${RIVERMAX_DEV_KIT_DOXYFILE_INPUT_LIST}
    MAIN_DEPENDENCY ${RIVERMAX_DEV_KIT_DOXYFILE_OUT}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${RIVERMAX_DEV_KIT_DOXYGEN_HTML_DIR}/docs/images
    COMMAND ${CMAKE_COMMAND} -E copy
        ${RIVERMAX_DEV_KIT_SW_STACK_IMAGE_PATH}
        ${RIVERMAX_DEV_KIT_DOXYGEN_HTML_DIR}/docs/images/sw_stack.png
    COMMAND Doxygen::doxygen ${RIVERMAX_DEV_KIT_DOXYFILE_OUT}
    VERBATIM
)

# Add a custom target to generate the documentation:
add_custom_target(${PROJECT_NAME} DEPENDS ${RIVERMAX_DEV_KIT_DOXYGEN_INDEX_FILE})

# Exclude the target from all builds:
set_target_properties(${PROJECT_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
