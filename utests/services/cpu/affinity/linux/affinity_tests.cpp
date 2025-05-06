/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <array>
#include <cstddef>
#include <cstring>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "rdk/services/cpu/affinity/rivermax_affinity.h"

using namespace rivermax::dev_kit::services;
using namespace testing;

struct OsApiMock: public LinuxAffinity::os_api {
    MOCK_METHOD(cpu_set_t *, cpu_alloc, (size_t count), (const, override));
    MOCK_METHOD(void, cpu_free, (cpu_set_t *cpu_set), (const, override));
    MOCK_METHOD(size_t, cpu_alloc_size, (size_t count), (const, override));
    MOCK_METHOD(void, cpu_zero_s, (size_t size, cpu_set_t *cpu_set), (const, override));

    MOCK_METHOD(int, get_proc_count, (), (const, override));
    MOCK_METHOD(void, cpu_set, (size_t processor, cpu_set_t *cpu_set), (const, override));
    MOCK_METHOD(int, set_affinity_np,
        (pthread_t handle, size_t set_size, const cpu_set_t *cpu_set),
        (const, override));

};

struct AffinityMaskMock: public Affinity::mask {
    AffinityMaskMock() { std::memset(rmax_bits, 0, sizeof(rmax_bits)); }
    auto set_cpu(size_t cpu) {
        constexpr auto entry_bit_size = sizeof(rmax_bits[0]) << 3;
        const auto entry = cpu / entry_bit_size;
        const auto bit_offset = cpu % entry_bit_size;
        rmax_bits[entry] |= (1ULL << bit_offset);
        return *this;
    }
};

class AffinityTest: public Test {
protected:
    std::thread m_thread = std::thread([](){});
    cpu_set_t m_fake_cpu_set;
    const size_t m_fake_set_size = sizeof(m_fake_cpu_set)*3;

    Expectation expect_cpu_set_allocation(OsApiMock &api_mock) {
        Expectation allocation =
            EXPECT_CALL(api_mock, cpu_alloc(RMAX_CPU_SETSIZE))
                .WillOnce(Return(&m_fake_cpu_set))
                .RetiresOnSaturation();
        EXPECT_CALL(api_mock, cpu_alloc_size(RMAX_CPU_SETSIZE))
            .WillOnce(Return(m_fake_set_size));
        Expectation initialization =
            EXPECT_CALL(api_mock, cpu_zero_s(m_fake_set_size, &m_fake_cpu_set))
                .Times(1)
                .After(allocation);
        return initialization;
    }

public:
    virtual ~AffinityTest() { m_thread.join(); }

};

TEST_F(AffinityTest, NumericInput_SingleCore) {
    OsApiMock api_mock;
    Affinity affinity(api_mock);

    const std::array processor_numbers = {0, RMAX_CPU_SETSIZE/3, RMAX_CPU_SETSIZE - 1};
    for (auto processor: processor_numbers) {
        Expectation allocation = expect_cpu_set_allocation(api_mock);
        Expectation configuration =
            EXPECT_CALL(api_mock, cpu_set(processor, &m_fake_cpu_set))
                .Times(1)
                .After(allocation);
        Expectation api_call =
            EXPECT_CALL(api_mock,
                    set_affinity_np(m_thread.native_handle(), m_fake_set_size, &m_fake_cpu_set))
                .After(configuration)
                .WillOnce(Return(0))
                .RetiresOnSaturation();
        EXPECT_CALL(api_mock, cpu_free(&m_fake_cpu_set))
            .Times(1)
            .After(api_call)
            .RetiresOnSaturation();

        affinity.set(m_thread, processor);
    }
}

TEST_F(AffinityTest, NumericInput_CurrentThread) {
    OsApiMock api_mock;
    Affinity affinity(api_mock);
    size_t processor = RMAX_NCPUBITS - 1;
    Expectation allocation = expect_cpu_set_allocation(api_mock);
    Expectation configuration =
        EXPECT_CALL(api_mock, cpu_set(processor, &m_fake_cpu_set))
            .Times(1)
            .After(allocation);
    pthread_t current_thread = pthread_self();
    Expectation api_call =
        EXPECT_CALL(api_mock,
                set_affinity_np(current_thread, m_fake_set_size, &m_fake_cpu_set))
            .After(configuration)
            .WillOnce(Return(0))
            .RetiresOnSaturation();
    EXPECT_CALL(api_mock, cpu_free(&m_fake_cpu_set))
        .Times(1)
        .After(api_call)
        .RetiresOnSaturation();

    affinity.set(processor);
}

TEST_F(AffinityTest, NumericInput_ExceptionUponSetAllocation) {
    StrictMock<OsApiMock> api_mock;
    Affinity affinity(api_mock);

    EXPECT_CALL(api_mock, cpu_alloc(RMAX_CPU_SETSIZE))
                .WillOnce(Return(nullptr));

    EXPECT_THROW( {
        affinity.set(m_thread, 0);
    }, std::runtime_error);
}

TEST_F(AffinityTest, NumericInput_ExceptionUponSetAffinity) {
    OsApiMock api_mock;
    Affinity affinity(api_mock);

    Expectation allocation = expect_cpu_set_allocation(api_mock);
    Expectation configuration =
        EXPECT_CALL(api_mock, cpu_set(0, &m_fake_cpu_set))
            .Times(1)
            .After(allocation);
    Expectation api_call =
        EXPECT_CALL(api_mock,
                set_affinity_np(m_thread.native_handle(), m_fake_set_size, &m_fake_cpu_set))
            .After(configuration)
            .WillOnce(Return(-1))
            .RetiresOnSaturation();
    EXPECT_CALL(api_mock, cpu_free(&m_fake_cpu_set))
        .Times(1)
        .After(api_call)
        .RetiresOnSaturation();

    EXPECT_THROW( {
        affinity.set(m_thread, 0);
    }, std::runtime_error);
}

TEST_F(AffinityTest, NumericInput_ExceptionUponIllegalCoreNumber) {
    OsApiMock api_mock;
    Affinity affinity(api_mock);
    constexpr size_t bad_number {RMAX_CPU_SETSIZE};

    ON_CALL(api_mock, cpu_alloc(_))
        .WillByDefault([&](size_t size) {
            EXPECT_CALL(api_mock, cpu_free(&m_fake_cpu_set))
                .Times(1)
                .RetiresOnSaturation();
            return &m_fake_cpu_set;
        });
    ON_CALL(api_mock, cpu_alloc_size(RMAX_CPU_SETSIZE))
        .WillByDefault(Return(m_fake_set_size));

    EXPECT_CALL(api_mock, cpu_set(0, _))
        .Times(0);
    EXPECT_CALL(api_mock, set_affinity_np(_, _, _))
        .Times(0);

    EXPECT_THROW( {
        affinity.set(m_thread, RMAX_CPU_SETSIZE);
    }, std::runtime_error);
}

TEST_F(AffinityTest, MaskInput_SingleCore) {
    OsApiMock api_mock;
    Affinity affinity(api_mock);
    constexpr const size_t processor { RMAX_NCPUBITS };
    const auto cpu_mask = AffinityMaskMock().set_cpu(processor);

    Expectation allocation = expect_cpu_set_allocation(api_mock);
    Expectation configuration =
        EXPECT_CALL(api_mock, cpu_set(processor, &m_fake_cpu_set))
            .Times(1)
            .After(allocation);
    Expectation api_call =
        EXPECT_CALL(api_mock,
                set_affinity_np(m_thread.native_handle(), m_fake_set_size, &m_fake_cpu_set))
            .After(configuration)
            .WillOnce(Return(0))
            .RetiresOnSaturation();
    EXPECT_CALL(api_mock, cpu_free(&m_fake_cpu_set))
        .Times(1)
        .After(api_call)
        .RetiresOnSaturation();

    affinity.set(m_thread, cpu_mask);
}

TEST_F(AffinityTest, MaskInput_ManyCores) {
    OsApiMock api_mock;
    Affinity affinity(api_mock);
    constexpr const size_t processors[] { 0, RMAX_CPU_SETSIZE/2, RMAX_CPU_SETSIZE-1};
    const auto cpu_mask = AffinityMaskMock().set_cpu(processors[0]).set_cpu(processors[1]).set_cpu(processors[2]);

    Expectation allocation = expect_cpu_set_allocation(api_mock);
    ExpectationSet configuration;
    for (auto processor: processors) {
        configuration += EXPECT_CALL(api_mock, cpu_set(processor, &m_fake_cpu_set))
            .Times(1)
            .After(allocation);
    }
    Expectation api_call =
        EXPECT_CALL(api_mock,
                set_affinity_np(m_thread.native_handle(), m_fake_set_size, &m_fake_cpu_set))
            .After(configuration)
            .WillOnce(Return(0))
            .RetiresOnSaturation();
    EXPECT_CALL(api_mock, cpu_free(&m_fake_cpu_set))
        .Times(1)
        .After(api_call)
        .RetiresOnSaturation();

    affinity.set(m_thread, cpu_mask);
}

TEST_F(AffinityTest, MaskInput_CurrentThread) {
    OsApiMock api_mock;
    Affinity affinity(api_mock);
    constexpr const size_t processors[] { 0, RMAX_CPU_SETSIZE/2, RMAX_CPU_SETSIZE-1};
    const auto cpu_mask = AffinityMaskMock().set_cpu(processors[0]).set_cpu(processors[1]).set_cpu(processors[2]);

    Expectation allocation = expect_cpu_set_allocation(api_mock);
    ExpectationSet configuration;
    for (auto processor: processors) {
        configuration += EXPECT_CALL(api_mock, cpu_set(processor, &m_fake_cpu_set))
            .Times(1)
            .After(allocation);
    }
    pthread_t current_thread = pthread_self();
    Expectation api_call =
        EXPECT_CALL(api_mock,
                set_affinity_np(current_thread, m_fake_set_size, &m_fake_cpu_set))
            .After(configuration)
            .WillOnce(Return(0))
            .RetiresOnSaturation();
    EXPECT_CALL(api_mock, cpu_free(&m_fake_cpu_set))
        .Times(1)
        .After(api_call)
        .RetiresOnSaturation();

    affinity.set(cpu_mask);
}

TEST_F(AffinityTest, MaskInput_ExceptionUponSetAllocation) {
    StrictMock<OsApiMock> api_mock;
    Affinity affinity(api_mock);
    const auto cpu_mask = AffinityMaskMock().set_cpu(0);

    EXPECT_CALL(api_mock, cpu_alloc(RMAX_CPU_SETSIZE))
                .WillOnce(Return(nullptr));

    EXPECT_THROW( {
        affinity.set(m_thread, cpu_mask);
    }, std::runtime_error);
}

TEST_F(AffinityTest, MaskInput_ExceptionUponSetAffinity) {
    OsApiMock api_mock;
    Affinity affinity(api_mock);
    const auto cpu_mask = AffinityMaskMock().set_cpu(0);

    Expectation allocation = expect_cpu_set_allocation(api_mock);
    Expectation configuration =
        EXPECT_CALL(api_mock, cpu_set(0, &m_fake_cpu_set))
            .Times(1)
            .After(allocation);
    Expectation api_call =
        EXPECT_CALL(api_mock,
                set_affinity_np(m_thread.native_handle(), m_fake_set_size, &m_fake_cpu_set))
            .After(configuration)
            .WillOnce(Return(-1))
            .RetiresOnSaturation();
    EXPECT_CALL(api_mock, cpu_free(&m_fake_cpu_set))
        .Times(1)
        .After(api_call)
        .RetiresOnSaturation();

    EXPECT_THROW( {
        affinity.set(m_thread, cpu_mask);
    }, std::runtime_error);
}

TEST_F(AffinityTest, MaskInput_EmptyBitmapOverGivenThread) {
    OsApiMock api_mock;
    Affinity affinity(api_mock);
    Affinity::mask affinity_mask;
    memset(&affinity_mask, 0, sizeof(affinity_mask));

    expect_cpu_set_allocation(api_mock);

    EXPECT_THROW( {
        affinity.set(m_thread, affinity_mask);
    }, std::underflow_error);
}

TEST_F(AffinityTest, MaskInput_EmptyBitmapOverCurrentThread) {
    OsApiMock api_mock;
    Affinity affinity(api_mock);
    Affinity::mask affinity_mask;
    memset(&affinity_mask, 0, sizeof(affinity_mask));

    expect_cpu_set_allocation(api_mock);

    EXPECT_THROW( {
        affinity.set(affinity_mask);
    }, std::underflow_error);
}

TEST_F(AffinityTest, CoreCount_Single) {
    OsApiMock api_mock;
    Affinity affinity(api_mock);

    EXPECT_CALL(api_mock, get_proc_count())
        .WillOnce(Return(1));

    ASSERT_EQ(1, affinity.count_cores());
}

TEST_F(AffinityTest, CoreCount_Many)
{
    OsApiMock api_mock;
    Affinity affinity(api_mock);

    EXPECT_CALL(api_mock, get_proc_count())
        .WillOnce(Return(63));

    ASSERT_EQ(63, affinity.count_cores());
}