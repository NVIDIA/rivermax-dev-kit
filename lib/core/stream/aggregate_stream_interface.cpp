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
