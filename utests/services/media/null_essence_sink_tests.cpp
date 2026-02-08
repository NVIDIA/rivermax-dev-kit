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

#include <memory>

#include <gtest/gtest.h>

#include "rdk/services/media/null_essence_sink.h"
#include "rdk/services/media/media_defs.h"

using namespace rivermax::dev_kit::services;

/* Put-failure probability constants [0, 1] */
static constexpr float PROBABILITY_ZERO = 0.0f;
static constexpr float PROBABILITY_ONE = 1.0f;
static constexpr float PROBABILITY_INVALID_BELOW = -0.1f;
static constexpr float PROBABILITY_INVALID_ABOVE = 1.1f;
static constexpr float PROBABILITY_MID_VALID = 0.5f;

/**
 * @brief: Test fixture for NullEssenceSink.
 */
class NullEssenceSinkTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_sink = std::make_unique<NullEssenceSink>();
        m_media_unit = std::make_shared<MediaUnit>(1024, SMPTEStandard::ST_2110_20);
    }

    void TearDown() override
    {
        m_media_unit.reset();
        m_sink.reset();
    }

    std::unique_ptr<NullEssenceSink> m_sink;
    std::shared_ptr<MediaUnit> m_media_unit;
};

/* Test: default constructor initializes put-failure probability to zero */
TEST_F(NullEssenceSinkTest, Constructor_ZeroFailureProbability)
{
    EXPECT_FLOAT_EQ(m_sink->get_put_failure_probability(), PROBABILITY_ZERO);
}

/* Test: put_media_unit_blocking always succeeds regardless of probability */
TEST_F(NullEssenceSinkTest, PutBlocking_AlwaysSucceeds)
{
    ASSERT_EQ(m_sink->set_put_failure_probability(PROBABILITY_ONE), ReturnStatus::success);
    EXPECT_EQ(m_sink->put_media_unit_blocking(m_media_unit), ReturnStatus::success);
}

/* Test: put_media_unit_non_blocking respects probability */
TEST_F(NullEssenceSinkTest, PutNonBlocking_RespectsProbability)
{
    m_sink->set_put_failure_probability(PROBABILITY_ZERO);
    EXPECT_EQ(m_sink->put_media_unit_non_blocking(m_media_unit), ReturnStatus::success);
    m_sink->set_put_failure_probability(PROBABILITY_ONE);
    EXPECT_EQ(m_sink->put_media_unit_non_blocking(m_media_unit), ReturnStatus::sink_not_ready);
}

/* Test: set_put_failure_probability rejects out-of-range and accepts valid */
TEST_F(NullEssenceSinkTest, SetPutFailureProbability_ValidAndInvalid)
{
    EXPECT_EQ(m_sink->set_put_failure_probability(PROBABILITY_INVALID_BELOW), ReturnStatus::failure);
    EXPECT_FLOAT_EQ(m_sink->get_put_failure_probability(), PROBABILITY_ZERO);
    EXPECT_EQ(m_sink->set_put_failure_probability(PROBABILITY_INVALID_ABOVE), ReturnStatus::failure);
    EXPECT_FLOAT_EQ(m_sink->get_put_failure_probability(), PROBABILITY_ZERO);
    EXPECT_EQ(m_sink->set_put_failure_probability(PROBABILITY_MID_VALID), ReturnStatus::success);
    EXPECT_FLOAT_EQ(m_sink->get_put_failure_probability(), PROBABILITY_MID_VALID);
}

/* Test: stop is no-op */
TEST_F(NullEssenceSinkTest, Stop_NoOp)
{
    EXPECT_NO_THROW(m_sink->stop());
    EXPECT_EQ(m_sink->put_media_unit_blocking(m_media_unit), ReturnStatus::success);
}

/* Test: put_media_unit with nullptr returns failure */
TEST_F(NullEssenceSinkTest, Put_NullptrReturnsFailure)
{
    EXPECT_EQ(m_sink->put_media_unit_blocking(nullptr), ReturnStatus::failure);
    EXPECT_EQ(m_sink->put_media_unit_non_blocking(nullptr), ReturnStatus::failure);
}

/* Test: put_media_unit_blocking accepts all SMPTE standards */
TEST_F(NullEssenceSinkTest, PutBlocking_AcceptsAllSMPTEStandards)
{
    EXPECT_EQ(m_sink->put_media_unit_blocking(
                  std::make_shared<MediaUnit>(1024, SMPTEStandard::ST_2110_20)),
              ReturnStatus::success);
    EXPECT_EQ(m_sink->put_media_unit_blocking(
                  std::make_shared<MediaUnit>(256, SMPTEStandard::ST_2110_30)),
              ReturnStatus::success);
    EXPECT_EQ(m_sink->put_media_unit_blocking(
                  std::make_shared<MediaUnit>(512, SMPTEStandard::ST_2110_40)),
              ReturnStatus::success);
}
