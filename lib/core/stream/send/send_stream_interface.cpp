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

#include <ostream>

#include "core/stream/send/send_stream_interface.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

ral::lib::core::ISendStream::ISendStream(const ral::lib::core::TwoTupleFlow& local_address) :
    ISingleStream(local_address),
    m_num_of_chunks(0)
{
}

std::ostream& ral::lib::core::ISendStream::print(std::ostream& out) const
{
    return ISingleStream::print(out)
        << "| Number of chunks: " << m_num_of_chunks << "\n";
}
