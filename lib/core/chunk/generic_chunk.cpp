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

#include <cstddef>
#include <memory>

#include <rivermax_api.h>

#include "core/chunk/generic_chunk.h"

using namespace ral::lib::core;

GenericChunk::GenericChunk() :
    m_rmax_chunk(new rmax_chunk)
{
}

GenericChunk::GenericChunk(rmax_chunk* _rmax_chunk) :
    m_rmax_chunk(_rmax_chunk)
{
}
