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

#include "core/stream/single_stream_interface.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

ral::lib::core::ISingleStream::ISingleStream(const ral::lib::core::TwoTupleFlow& local_address) :
    IStream(),
    m_local_address(local_address),
    m_stream_id(0),
    m_stream_id_set(false)
{
}

std::ostream& ral::lib::core::ISingleStream::print(std::ostream& out) const
{
    out << "| Stream ID: ";
    if (m_stream_id_set) {
        out << m_stream_id << "\n";
    } else {
        out << "not set\n";
    }
    out << "| NIC IP: " << m_local_address.get_ip() << "\n";
        // TODO: Enable this after adding set and/or query source port support.
        //<< "| Source port: " << m_local_address.get_source_port() << "\n";

    return out;
}
