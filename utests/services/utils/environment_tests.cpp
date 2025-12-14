/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdlib>

#include <gtest/gtest.h>

#include "rdk/services/utils/environment.h"

using namespace rivermax::dev_kit::services;

/**
 * @brief: Test fixture for EnvironmentUtils class.
 *
 * Provides setup and teardown for environment variable tests,
 * ensuring test variables are cleaned up after each test.
 */
class EnvironmentUtilsTests : public ::testing::Test
{
protected:
    const std::string TEST_VAR_NAME = "RDK_TEST_ENV_VAR_12345";

    void TearDown() override
    {
        // Clean up test variable after each test
#ifdef __linux__
        unsetenv(TEST_VAR_NAME.c_str());
#elif defined(_WIN32)
        _putenv_s(TEST_VAR_NAME.c_str(), "");
#endif
    }
};

/* Test setting and retrieving an environment variable. */
TEST_F(EnvironmentUtilsTests, SetAndGet)
{
    const std::string test_value = "test_value_123";

    ReturnStatus set_status = EnvironmentUtils::set_variable(TEST_VAR_NAME, test_value);
    ASSERT_EQ(set_status, ReturnStatus::success);

    std::string retrieved_value;
    ReturnStatus get_status = EnvironmentUtils::get_variable(TEST_VAR_NAME, retrieved_value);
    ASSERT_EQ(get_status, ReturnStatus::success);
    ASSERT_EQ(retrieved_value, test_value);
}

/* Test that retrieving a non-existent environment variable returns failure. */
TEST_F(EnvironmentUtilsTests, GetNonExistent)
{
    std::string value;
    ReturnStatus status = EnvironmentUtils::get_variable("RDK_NON_EXISTENT_VAR_99999", value);
    ASSERT_EQ(status, ReturnStatus::failure);
}

/* Test that setting an environment variable overwrites the previous value. */
TEST_F(EnvironmentUtilsTests, SetOverwrite)
{
    const std::string initial_value = "initial";
    const std::string new_value = "updated";

    ASSERT_EQ(EnvironmentUtils::set_variable(TEST_VAR_NAME, initial_value), ReturnStatus::success);

    std::string retrieved;
    ASSERT_EQ(EnvironmentUtils::get_variable(TEST_VAR_NAME, retrieved), ReturnStatus::success);
    ASSERT_EQ(retrieved, initial_value);

    ASSERT_EQ(EnvironmentUtils::set_variable(TEST_VAR_NAME, new_value), ReturnStatus::success);

    ASSERT_EQ(EnvironmentUtils::get_variable(TEST_VAR_NAME, retrieved), ReturnStatus::success);
    ASSERT_EQ(retrieved, new_value);
}

/* Test setting an environment variable to an empty string. */
TEST_F(EnvironmentUtilsTests, SetEmptyValue)
{
    ASSERT_EQ(EnvironmentUtils::set_variable(TEST_VAR_NAME, ""), ReturnStatus::success);

    std::string retrieved;
    ReturnStatus status = EnvironmentUtils::get_variable(TEST_VAR_NAME, retrieved);
    // On Linux, empty string is valid; on Windows, it may unset the variable
#ifdef __linux__
    ASSERT_EQ(status, ReturnStatus::success);
    ASSERT_EQ(retrieved, "");
#endif
}

/* Test setting an environment variable with special characters in the value. */
TEST_F(EnvironmentUtilsTests, SetValueWithSpecialCharacters)
{
    const std::string special_value = "path/to/file:with=special;chars";

    ASSERT_EQ(EnvironmentUtils::set_variable(TEST_VAR_NAME, special_value), ReturnStatus::success);

    std::string retrieved;
    ASSERT_EQ(EnvironmentUtils::get_variable(TEST_VAR_NAME, retrieved), ReturnStatus::success);
    ASSERT_EQ(retrieved, special_value);
}
