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

#include <bitset>
#include <stdexcept>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "rdk/services/cpu/affinity/rivermax_affinity.h"

using namespace rivermax::dev_kit::services;
using namespace testing;

struct WinApiMock: public WindowsAffinity::os_api {
    MOCK_METHOD(DWORD, get_logical_processor_information_ex,
        (const LOGICAL_PROCESSOR_RELATIONSHIP RelationshipType, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *Buffer, DWORD *ReturnedLength),
        (const, override));

    MOCK_METHOD(DWORD, set_thread_group_affinity,
        (HANDLE hThread, const GROUP_AFFINITY *GroupAffinity, GROUP_AFFINITY *PreviousGroupAffinity), (const, override));
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

template <size_t PROCESSOR_GROUPS>
struct ProcessorGroupInformationMock: public SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX {
    static constexpr BYTE MAX_PROCESSOR_COUNT {64};
    ProcessorGroupInformationMock(const KAFFINITY (&masks)[PROCESSOR_GROUPS])
        : ProcessorGroupInformationMock()
    {
        for (auto i = 0; i < PROCESSOR_GROUPS; ++i) {
            PROCESSOR_GROUP_INFO &group = Group.GroupInfo[i];
            group.ActiveProcessorMask = masks[i];
            group.MaximumProcessorCount = MAX_PROCESSOR_COUNT;
            group.ActiveProcessorCount = (BYTE)std::bitset<MAX_PROCESSOR_COUNT>(group.ActiveProcessorMask).count();
        }
    }
protected:
    PROCESSOR_GROUP_INFO _[PROCESSOR_GROUPS];
    ProcessorGroupInformationMock()
    {
        std::memset(this, 0, sizeof(*this));
        Relationship = RelationGroup;
        Size = sizeof(*this);
        Group.MaximumGroupCount = PROCESSOR_GROUPS;
        Group.ActiveGroupCount = PROCESSOR_GROUPS;
    }
};

class AffinityTest: public Test {
protected:
    StrictMock<WinApiMock> m_api_mock;
    std::thread m_thread = std::thread([](){});

public:
    virtual ~AffinityTest() { m_thread.join(); }

    template <int PROCESSOR_GROUPS>
    Affinity setup_affinity(const KAFFINITY (&processor_group_masks)[PROCESSOR_GROUPS]) {
        ProcessorGroupInformationMock info_mock {processor_group_masks};
        DWORD size = (DWORD)sizeof(info_mock);

        EXPECT_CALL(m_api_mock, get_logical_processor_information_ex(
                RelationGroup,
                _,
                Pointee(0)))
            .WillOnce(DoAll(SetArgPointee<2>(size),Return(ERROR_INSUFFICIENT_BUFFER)));

        EXPECT_CALL(m_api_mock, get_logical_processor_information_ex(
                RelationGroup,
                NotNull(),
                Pointee((DWORD)sizeof(info_mock))))
            .WillOnce([&]([[maybe_unused]] const LOGICAL_PROCESSOR_RELATIONSHIP type,
                SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *buffer, DWORD *length){
                    std::memcpy(buffer, &info_mock, *length);
                    return NO_ERROR;
            });

        return Affinity(m_api_mock);
    }
};

TEST_F(AffinityTest, NumericInput_SingleCore) {
    const KAFFINITY masks[1] {0b1};
    Affinity affinity = setup_affinity(masks);

    constexpr size_t processor {0};
    constexpr KAFFINITY expected_mask {1 << processor};
    EXPECT_CALL(m_api_mock, set_thread_group_affinity(
            m_thread.native_handle(),
            AllOf(
                NotNull(),
                Pointee(Field(&GROUP_AFFINITY::Mask, expected_mask)),
                Pointee(Field(&GROUP_AFFINITY::Group,(WORD)0))
            ),
            nullptr))
        .WillOnce(Return(NO_ERROR));

    affinity.set(m_thread, processor);
}

TEST_F(AffinityTest, NumericInput_ManyCoresSingleGroup) {
    KAFFINITY masks[] {0b11110001100101};
    Affinity affinity = setup_affinity(masks);
    size_t processor = 0;

    for (size_t i = 0; (1ULL << i) < masks[0]; ++i) {
        if (((1ULL << i) & masks[0]) == 0) continue;

        const KAFFINITY expected_mask {1ULL << i};
        EXPECT_CALL(m_api_mock, set_thread_group_affinity(
                m_thread.native_handle(),
                AllOf(
                    NotNull(),
                    Pointee(Field(&GROUP_AFFINITY::Mask, expected_mask)),
                    Pointee(Field(&GROUP_AFFINITY::Group,(WORD)0))
                ),
                nullptr))
            .WillOnce(Return(NO_ERROR));

        affinity.set(m_thread, processor++);
    }
}

TEST_F(AffinityTest, NumericInput_ManyGroups) {
    KAFFINITY masks[2] {0b11110001100101, 0b1111110011};
    Affinity affinity = setup_affinity(masks);

    constexpr size_t processor {12};
    constexpr KAFFINITY expected_mask {0b1000000};
    EXPECT_CALL(m_api_mock, set_thread_group_affinity(
            m_thread.native_handle(),
            AllOf(
                NotNull(),
                Pointee(Field(&GROUP_AFFINITY::Mask, expected_mask)),
                Pointee(Field(&GROUP_AFFINITY::Group,(WORD)1))
            ),
            nullptr))
        .WillOnce(Return(NO_ERROR));

    affinity.set(m_thread, processor);
}

TEST_F(AffinityTest, NumericInput_CoreHighNumber) {
    constexpr KAFFINITY expected_mask {(1ULL << 63)};
    KAFFINITY masks[2] {0b11110001100101, 0b111110011 | expected_mask};
    Affinity affinity = setup_affinity(masks);

    constexpr size_t processor {15};
    EXPECT_CALL(m_api_mock, set_thread_group_affinity(
            m_thread.native_handle(),
            AllOf(
                NotNull(),
                Pointee(Field(&GROUP_AFFINITY::Mask, expected_mask)),
                Pointee(Field(&GROUP_AFFINITY::Group,(WORD)1))
            ),
            nullptr))
        .WillOnce(Return(NO_ERROR));

    affinity.set(m_thread, processor);
}

TEST_F(AffinityTest, NumericInput_CurrentThread) {
    KAFFINITY masks[] {0b11110001100101};
    Affinity affinity = setup_affinity(masks);
    size_t processor = 0;
    auto thread_handle = ::GetCurrentThread();

    for (size_t i = 0; (1ULL << i) < masks[0]; ++i) {
        if (((1ULL << i) & masks[0]) == 0) continue;

        const KAFFINITY expected_mask {1ULL << i};
        EXPECT_CALL(m_api_mock, set_thread_group_affinity(
                thread_handle,
                AllOf(
                    NotNull(),
                    Pointee(Field(&GROUP_AFFINITY::Mask, expected_mask)),
                    Pointee(Field(&GROUP_AFFINITY::Group,(WORD)0))
                ),
                nullptr))
            .WillOnce(Return(NO_ERROR));

        affinity.set(processor++);
    }
}

TEST_F(AffinityTest, Init_ExceptionUponSizeRequest) {
    const KAFFINITY masks[] {0b1};
    ProcessorGroupInformationMock info_mock {masks};
    DWORD size = (DWORD)sizeof(info_mock);

    EXPECT_CALL(m_api_mock, get_logical_processor_information_ex(_,_,_))
        .WillOnce(DoAll(SetArgPointee<2>(size),Return(ERROR_ACCESS_DENIED)));

    EXPECT_THROW( {
        Affinity affinity(m_api_mock);
    }, std::runtime_error);
}

TEST_F(AffinityTest, Init_ExceptionUponDatabaseRequest) {
    const KAFFINITY masks[] {0b1};
    ProcessorGroupInformationMock info_mock {masks};
    DWORD size = (DWORD)sizeof(info_mock);

    EXPECT_CALL(m_api_mock, get_logical_processor_information_ex(_,_,_))
        .WillOnce(DoAll(SetArgPointee<2>(size),Return(ERROR_INSUFFICIENT_BUFFER)))
        .WillOnce(DoAll(SetArgPointee<2>(size),Return(ERROR_ACCESS_DENIED)));

    EXPECT_THROW( {
        Affinity affinity(m_api_mock);
    }, std::runtime_error);
}

TEST_F(AffinityTest, NumericInput_IllegalCoreNumber) {
    constexpr auto last_core_number = 0;
    constexpr auto affinity_max_mask = (1 << (last_core_number + 1)) - 1;
    const KAFFINITY masks[] {affinity_max_mask};
    Affinity affinity = setup_affinity(masks);

    constexpr size_t illegal_number {last_core_number + 1};

    EXPECT_THROW( {
        affinity.set(m_thread, illegal_number);
    }, std::runtime_error);
}

TEST_F(AffinityTest, NumericInput_ExceptionUponSettingAffinity) {
    const KAFFINITY masks[] {0b1};
    Affinity affinity = setup_affinity(masks);

    constexpr size_t processor {0};
    EXPECT_CALL(m_api_mock, set_thread_group_affinity(_,_,_))
        .WillOnce(Return(ERROR_ACCESS_DENIED));

    EXPECT_THROW( {
        affinity.set(m_thread, processor);
    }, std::runtime_error);
}

TEST_F(AffinityTest, MaskInput_ManyCoresSingleGroup) {
    KAFFINITY masks[] {0b11110001100101};
    Affinity affinity = setup_affinity(masks);

    constexpr Affinity::mask cpu_mask { 1 << 1 | 1 << 5 | 1 << 7};
    constexpr KAFFINITY expected_mask {0b10100000000100};
    EXPECT_CALL(m_api_mock, set_thread_group_affinity(
            m_thread.native_handle(),
            AllOf(
                NotNull(),
                Pointee(Field(&GROUP_AFFINITY::Mask, expected_mask)),
                Pointee(Field(&GROUP_AFFINITY::Group,(WORD)0))
            ),
            nullptr))
        .WillOnce(Return(NO_ERROR));

    affinity.set(m_thread, cpu_mask);
}

TEST_F(AffinityTest, MaskInput_ManyCoresManyGroups) {
    KAFFINITY masks[2] {0b11110001100101, 0b1111110011};
    Affinity affinity = setup_affinity(masks);
    constexpr Affinity::mask cpu_mask { 1 << 1 | 1 << 5 | 1 << 12 };

    constexpr KAFFINITY expected_mask_0 {0b100000000100};
    EXPECT_CALL(m_api_mock, set_thread_group_affinity(
            m_thread.native_handle(),
            AllOf(
                NotNull(),
                Pointee(Field(&GROUP_AFFINITY::Mask, expected_mask_0)),
                Pointee(Field(&GROUP_AFFINITY::Group,(WORD)0))
            ),
            nullptr))
        .WillOnce(Return(NO_ERROR));

    constexpr KAFFINITY expected_mask_1 {0b1000000};
    EXPECT_CALL(m_api_mock, set_thread_group_affinity(
            m_thread.native_handle(),
            AllOf(
                NotNull(),
                Pointee(Field(&GROUP_AFFINITY::Mask, expected_mask_1)),
                Pointee(Field(&GROUP_AFFINITY::Group,(WORD)1))
            ),
            nullptr))
        .WillOnce(Return(NO_ERROR));

    affinity.set(m_thread, cpu_mask);
}

TEST_F(AffinityTest, MaskInput_CurrentThread) {
    KAFFINITY masks[2] {0b11110001100101, 0b1111110011};
    Affinity affinity = setup_affinity(masks);
    constexpr Affinity::mask cpu_mask { 1 << 1 | 1 << 5 | 1 << 12 };

    auto thread_handle = ::GetCurrentThread();
    constexpr KAFFINITY expected_mask_0 {0b100000000100};
    EXPECT_CALL(m_api_mock, set_thread_group_affinity(
            thread_handle,
            AllOf(
                NotNull(),
                Pointee(Field(&GROUP_AFFINITY::Mask, expected_mask_0)),
                Pointee(Field(&GROUP_AFFINITY::Group,(WORD)0))
            ),
            nullptr))
        .WillOnce(Return(NO_ERROR));

    constexpr KAFFINITY expected_mask_1 {0b1000000};
    EXPECT_CALL(m_api_mock, set_thread_group_affinity(
            thread_handle,
            AllOf(
                NotNull(),
                Pointee(Field(&GROUP_AFFINITY::Mask, expected_mask_1)),
                Pointee(Field(&GROUP_AFFINITY::Group,(WORD)1))
            ),
            nullptr))
        .WillOnce(Return(NO_ERROR));

    affinity.set(cpu_mask);
}

TEST_F(AffinityTest, MaskInput_IllegalCoreNumber) {
    constexpr auto last_core_number = 3;
    constexpr auto affinity_max_mask = (1 << (last_core_number + 1)) - 1;
    constexpr KAFFINITY masks[] {affinity_max_mask};
    Affinity affinity = setup_affinity(masks);
    constexpr Affinity::mask illegal_mask { 1 << (last_core_number + 1)};

    EXPECT_THROW( {
        affinity.set(m_thread, illegal_mask);
    }, std::runtime_error);
}

TEST_F(AffinityTest, MaskInput_ExceptionUponSettingAffinity) {
    constexpr size_t processor {7};
    constexpr KAFFINITY masks[] {(1 << (processor + 1)) - 1};
    Affinity affinity = setup_affinity(masks);
    constexpr Affinity::mask cpu_mask {1 << processor};

    EXPECT_CALL(m_api_mock, set_thread_group_affinity(_,_,_))
        .WillOnce(Return(ERROR_ACCESS_DENIED));

    EXPECT_THROW( {
        affinity.set(m_thread, cpu_mask);
    }, std::runtime_error);
}

TEST_F(AffinityTest, MaskInput_EmptyBitmapOverGivenThread) {
    constexpr KAFFINITY masks[] {0};
    Affinity affinity = setup_affinity(masks);
    constexpr Affinity::mask affinity_mask { 0};

    EXPECT_THROW( {
        affinity.set(m_thread, affinity_mask);
    }, std::underflow_error);
}

TEST_F(AffinityTest, MaskInput_EmptyBitmapOverCurrentThread) {
    constexpr KAFFINITY masks[] {0};
    Affinity affinity = setup_affinity(masks);
    constexpr Affinity::mask affinity_mask { 0};

    EXPECT_THROW( {
        affinity.set(affinity_mask);
    }, std::underflow_error);
}

TEST_F(AffinityTest, CoreCount_Single) {
    const KAFFINITY masks[1] {0b1};
    Affinity affinity = setup_affinity(masks);

    ASSERT_EQ(1, affinity.count_cores());
}

TEST_F(AffinityTest, CoreCount_Many) {
    KAFFINITY masks[2] {0b11110001100101, 0b1111110011};
    Affinity affinity = setup_affinity(masks);
    auto expected =
        std::bitset<64>(masks[0]).count() +
        std::bitset<64>(masks[1]).count();

    ASSERT_EQ(expected, affinity.count_cores());
}