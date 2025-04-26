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
