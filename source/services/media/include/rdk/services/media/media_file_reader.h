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

#ifndef RDK_SERVICES_MEDIA_MEDIA_FILE_READER_H_
#define RDK_SERVICES_MEDIA_MEDIA_FILE_READER_H_

#include <fstream>
#include <string>

#include "rdk/services/error_handling/error_handling.h"
#include "rdk/services/memory_allocation/memory_allocation.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Unified file reader for media data.
 *
 * This class encapsulates common file reading operations with support for looping
 * when the end of file is reached. It uses @ref MemoryUtils for memory operations.
 */
class MediaFileReader
{
protected:
    std::string m_file_path;
    std::ifstream m_input_file;
    std::shared_ptr<MemoryUtils> m_memory_utils;
    bool m_loop_on_eof;
    size_t m_file_size;
    /**
     * @brief: Check if file is open.
     */
    bool is_file_open() const { return m_input_file.is_open(); }

public:
    /**
     * @brief: Constructor.
     *
     * @param [in] file_path: Path to the media file (can be empty).
     * @param [in] memory_utils: Memory utilities for buffer operations.
     * @param [in] loop_on_eof: Whether to loop back to start when EOF is reached.
     */
    MediaFileReader(
        const std::string& file_path,
        const std::shared_ptr<MemoryUtils>& memory_utils,
        bool loop_on_eof = false);
    virtual ~MediaFileReader();
    /**
     * @brief: Open the file for reading.
     *
     * @return: Status of the operation.
     */
    ReturnStatus open();
    /**
     * @brief: Close the file.
     */
    void close();
    /**
     * @brief: Get file size.
     *
     * @param [out] file_size: Size of the file in bytes.
     *
     * @return: Status of the operation.
     */
    ReturnStatus get_file_size(size_t& file_size) const;
    /**
     * @brief: Read a chunk of data from the file.
     *
     * @param [out] buffer: Buffer to read data into.
     * @param [in] size: Number of bytes to read.
     * @param [out] bytes_read: Actual number of bytes read.
     *
     * @return: Status of the operation.
     */
    ReturnStatus read_chunk(byte_t* buffer, size_t size, size_t& bytes_read);
    /**
     * @brief: Read data and copy to destination using MemoryUtils.
     *
     * Reads data from file and copies it to the destination buffer. If the read
     * is partial (less than requested size) and pad_partial is true, the remaining
     * bytes are filled with zeros.
     *
     * @param [out] dest_buffer: Destination buffer.
     * @param [in] size: Number of bytes to read.
     * @param [out] bytes_read: Actual number of bytes read (before padding).
     * @param [in] pad_partial: If true, pad partial reads with zeros to reach size.
     *
     * @return: Status of the operation.
     */
    ReturnStatus read_and_copy(byte_t* dest_buffer, size_t size, size_t& bytes_read, bool pad_partial = false);
    /**
     * @brief: Reset file to beginning.
     */
    void reset();
    /**
     * @brief: Check if EOF reached.
     */
    bool is_eof() const { return m_input_file.eof(); }
    /**
     * @brief: Handle EOF based on loop setting.
     *
     * @return: true if should continue reading (looped), false otherwise.
     */
    bool handle_eof();
};

}  // namespace services
}  // namespace dev_kit
}  // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_MEDIA_FILE_READER_H_ */
