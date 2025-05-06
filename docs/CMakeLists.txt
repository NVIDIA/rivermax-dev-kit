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

if (NOT TARGET Rivermax::Rivermax)
    message(WARNING "No Rivermax library was determined to generate the docs.")
    return()
endif()

find_package(Doxygen COMPONENTS dot)
if (NOT TARGET Doxygen::doxygen)
    message(STATUS "No Doxygen tool was determined to generate the docs.")
    return()
endif()

# Input files
set(RMAX_INPUT_FILES "rivermax_api.h" "rivermax_defs.h" "rivermax_deprecated.h" "rmx_stats_api.h" "rmx_stats_defs.h")
set(RMAX_APPS_LIB_INPUT_FILES ${PROJECT_SOURCE_DIR}/. ${PROJECT_SOURCE_DIR}/README.md)

get_target_property(RMAX_INTERFACE_INCLUDE Rivermax::Rivermax INTERFACE_INCLUDE_DIRECTORIES)
list(TRANSFORM RMAX_INPUT_FILES PREPEND ${RMAX_INTERFACE_INCLUDE}/)

string(REPLACE ";" "\n" DOXYGEN_INPUT "${RMAX_INPUT_FILES} ${RMAX_APPS_LIB_INPUT_FILES}")
set(DOXYFILE_IN         ${CMAKE_CURRENT_LIST_DIR}/Doxyfile.in)

set(DOXYGEN_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/docs)
set(DOXYFILE_OUT        ${PROJECT_BINARY_DIR}/Doxyfile)
set(DOXYGEN_INDEX_FILE  ${DOXYGEN_OUTPUT_DIR}/html/index.xml)

# Ensure the output folder for Doxygen
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

# Substitute variables inside the @@s with the correct values
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)


# Custom command to invoke Doxygen if headers or Doxyfile change
add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                   DEPENDS ${RMAX_INTERFACE_HEADERS}
                   COMMAND Doxygen::doxygen ${DOXYFILE_OUT}
                   MAIN_DEPENDENCY ${DOXYFILE_IN}
                   COMMENT "Generating documentation"
                   VERBATIM)

# Custom target to invoke the custom command specified above
add_custom_target(rmax_apps_lib_docs ALL DEPENDS ${DOXYGEN_INDEX_FILE})
set_target_properties(rmax_apps_lib_docs PROPERTIES EXCLUDE_FROM_ALL TRUE)
