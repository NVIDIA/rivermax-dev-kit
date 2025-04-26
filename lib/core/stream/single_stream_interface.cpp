/*
 * Copyright © 2017-2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#include "core/stream/single_stream_interface.h"

using namespace ral::lib::core;
using namespace ral::lib::services;

ral::lib::core::ISingleStream::ISingleStream(const ral::lib::core::TwoTupleFlow& local_address) :
    IStream(),
    m_local_address(local_address),
    m_stream_id(-1)
{
}

std::ostream& ral::lib::core::ISingleStream::print(std::ostream& out) const
{
    out << "| Stream ID: " << m_stream_id << "\n"
        << "| NIC IP: " << m_local_address.get_ip() << "\n";
        // TODO: Enable this after adding set and/or query source port support.
        //<< "| Source port: " << m_local_address.get_source_port() << "\n";

    return out;
}
