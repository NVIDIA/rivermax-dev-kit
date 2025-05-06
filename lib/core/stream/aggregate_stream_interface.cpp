/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/stream/aggregate_stream_interface.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

ral::lib::core::IAggregateStream::IAggregateStream(size_t id) :
    IStream(),
    m_stream_id(id)
{
}

std::ostream& ral::lib::core::IAggregateStream::print(std::ostream& out) const
{
    return IStream::print(out)
        << "| Aggregate stream ID: " << m_stream_id << "\n";
}
