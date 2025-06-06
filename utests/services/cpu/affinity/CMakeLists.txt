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

project(rivermax-dev-kit-utests-services-cpu-affinity)

include(GoogleTest)

add_executable(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE Rivermax::Include rivermax-dev-kit-utests-utils)

string(TOLOWER "${CMAKE_SYSTEM_NAME}" os_specific_dir)
add_subdirectory(${os_specific_dir})

gtest_discover_tests(${PROJECT_NAME})
