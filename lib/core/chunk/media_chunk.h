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

#ifndef RMAX_APPS_LIB_LIB_CORE_CHUNK_MEDIA_CHUNK_H_

#include <cstddef>
#include <string>

#include <rivermax_api.h>

#include "core/chunk/chunk_interface.h"

namespace ral
{
namespace lib
{
namespace core
{

/**
 * @brief: Media API chunk interface class.
 *
 * This interfaces indented to wrap Rivermax Media API chunk.
 */
class MediaChunk : public IChunk
{
private:
    void* m_data_ptr;
    uint16_t* m_data_size_arr;
    void* m_app_hdr_ptr;
    uint16_t* m_app_hdr_size_arr;
    size_t m_length;
public:
    /**
     * @brief: MediaChunk default constructor.
     */
    MediaChunk();
    /**
     * @brief: MediaChunk constructor.
     *
     * @param [in] data_ptr: Pointer to the data array.
     * @param [in] data_size_arr: Pointer to the data array sizes.
     * @param [in] length: Length in packets of the chunk.
     */
    MediaChunk(void** data_ptr, uint16_t* data_size_arr, size_t length);
    /**
     * @brief: MediaChunk constructor.
     *
     * @param [in] data_ptr: Pointer to the data array.
     * @param [in] data_size_arr: Pointer to the data sizes array.
     * @param [in] app_hdr_ptr: Pointer to the application header array.
     * @param [in] app_hdr_size_arr: Pointer to the application header sizes array.
     * @param [in] length: Length in packets of the chunk.
     */
    MediaChunk(void** data_ptr, uint16_t* data_size_arr,
               void** app_hdr_ptr, uint16_t* app_hdr_size_arr, size_t length);
    virtual size_t get_length() const override { return m_length; };
    /**
     * @brief: Sets the length of the chunk.
     *
     * @param [in] length: Length in packets of the chunk.
     */
    void set_length(size_t length) { m_length = length; };
    /**
     * @brief: Pointer to the underlay data array of the chunk.
     *
     * @returns: Pointer to the data array.
     */
    void** get_data_ptr() { return &m_data_ptr; };
    /**
     * @brief: Pointer to the underlay application header array of the chunk.
     *
     * @returns: Pointer to the application header array.
     */
    void** get_app_hdr_ptr() { return &m_app_hdr_ptr; };
    /**
     * @brief: Pointer to the underlay data sizes array of the chunk.
     *
     * @returns: Pointer to the data sizes array.
     */
    uint16_t* get_data_sizes_array() const { return m_data_size_arr; };
    /**
     * @brief: Pointer to the underlay application header sizes array of the chunk.
     *
     * @returns: Pointer to the application header sizes array.
     */
    uint16_t* get_app_hdr_sizes_array() const { return m_app_hdr_size_arr; };
};

} // namespace core
} // namespace lib
} // namespace ral

#define RMAX_APPS_LIB_LIB_CORE_CHUNK_MEDIA_CHUNK_H_
#endif /* RMAX_APPS_LIB_LIB_CORE_CHUNK_MEDIA_CHUNK_H_ */
