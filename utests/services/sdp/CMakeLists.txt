# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

project(rivermax-dev-kit-utests-services-sdp)

include(GoogleTest)

if (NOT WIN32)
    add_executable(${PROJECT_NAME})

    target_link_libraries(${PROJECT_NAME}
        PRIVATE
        rivermax-dev-kit-services
        rivermax-dev-kit-utests-utils
        GTest::gtest_main
        GTest::gmock_main
    )

    target_sources(${PROJECT_NAME}
        PRIVATE
        smpte_2110_20_tests.cpp
        smpte_2110_30_tests.cpp
        smpte_2110_40_tests.cpp
    )

    gtest_discover_tests(${PROJECT_NAME})
else()
    message(STATUS "Skipping SDP generator unit tests on Windows")
endif ()
