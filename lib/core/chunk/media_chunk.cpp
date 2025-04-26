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

#include "core/chunk/media_chunk.h"

using namespace ral::lib::core;

MediaChunk::MediaChunk() :
    m_data_ptr(nullptr),
    m_data_size_arr(nullptr),
    m_app_hdr_ptr(nullptr),
    m_app_hdr_size_arr(nullptr),
    m_length(0)
{
}

MediaChunk::MediaChunk(void** data_ptr, uint16_t* data_size_arr, size_t length) :
    m_data_ptr(data_ptr),
    m_data_size_arr(data_size_arr),
    m_app_hdr_ptr(nullptr),
    m_app_hdr_size_arr(nullptr),
    m_length(length)
{
}

MediaChunk::MediaChunk(void** data_ptr, uint16_t* data_size_arr,
                       void** app_hdr_ptr, uint16_t* app_hdr_size_arr, size_t length) :
    m_data_ptr(data_ptr),
    m_data_size_arr(data_size_arr),
    m_app_hdr_ptr(app_hdr_ptr),
    m_app_hdr_size_arr(app_hdr_size_arr),
    m_length(length)
{
}
