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

#include <sstream>

#include <gtest/gtest.h>

#include "rdk/services/utils/chrono_format.h"

using namespace std::chrono;

TEST(ChronoFormatTests, CustomRate)
{
    std::ostringstream oss;

    oss.str("");
    oss << duration<uint64_t, std::ratio<1, 90000>>(123);
    EXPECT_EQ(oss.str(), "123 [1/90000 s]");
}

TEST(ChronoFormatTests, Integer)
{
    std::ostringstream oss;

    oss.str("");
    oss << nanoseconds(987654321);
    EXPECT_EQ(oss.str(), "987654321 [ns]");

    oss.str("");
    oss << microseconds(456789);
    EXPECT_EQ(oss.str(), "456789 [us]");

    oss.str("");
    oss << milliseconds(123);
    EXPECT_EQ(oss.str(), "123 [ms]");

    oss.str("");
    oss << seconds(17);
    EXPECT_EQ(oss.str(), "17 [s]");

    oss.str("");
    oss << seconds(-17);
    EXPECT_EQ(oss.str(), "-17 [s]");

    oss.str("");
    oss << minutes(11);
    EXPECT_EQ(oss.str(), "11 [m]");

    oss.str("");
    oss << hours(8);
    EXPECT_EQ(oss.str(), "8 [h]");
}

TEST(ChronoFormatTests, Float)
{
    std::ostringstream oss;

    oss.str("");
    oss << duration<float, std::milli>(3.14);
    EXPECT_EQ(oss.str(), "3.14 [ms]");
}
