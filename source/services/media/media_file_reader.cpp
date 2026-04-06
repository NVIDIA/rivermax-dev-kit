/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "rdk/services/media/media_file_reader.h"

using namespace rdk::services;

MediaFileReader::MediaFileReader(
    const std::string& file_path,
    const std::shared_ptr<MemoryUtils>& memory_utils,
    bool loop_on_eof) :
    m_file_path(file_path),
    m_memory_utils(memory_utils),
    m_loop_on_eof(loop_on_eof),
    m_file_size(0)
{
}

MediaFileReader::~MediaFileReader()
{
    close();
}

ReturnStatus MediaFileReader::open()
{
    if (m_file_path.empty()) {
        std::cerr << "No file path provided" << std::endl;
        return ReturnStatus::failure;
    }

    if (m_input_file.is_open()) {
        return ReturnStatus::success;
    }

    m_input_file.open(m_file_path, std::ios::binary);
    if (!m_input_file.is_open()) {
        std::cerr << "Failed to open file: " << m_file_path << std::endl;
        return ReturnStatus::failure;
    }

    m_input_file.seekg(0, std::ios::end);
    m_file_size = m_input_file.tellg();
    m_input_file.seekg(0, std::ios::beg);
    m_input_file.clear();

    if (m_input_file.fail()) {
        std::cerr << "Failed to determine file size for: " << m_file_path << std::endl;
        return ReturnStatus::failure;
    }

    return ReturnStatus::success;
}

void MediaFileReader::close()
{
    if (m_input_file.is_open()) {
        m_input_file.close();
    }
}

ReturnStatus MediaFileReader::get_file_size(size_t& file_size) const
{
    file_size = m_file_size;
    return ReturnStatus::success;
}

ReturnStatus MediaFileReader::read_chunk(byte_t* buffer, size_t size, size_t& bytes_read)
{
    if (!m_input_file.is_open()) {
        bytes_read = 0;
        return ReturnStatus::success;  // No file is OK
    }

    m_input_file.read(reinterpret_cast<char*>(buffer), size);
    size_t count = static_cast<size_t>(m_input_file.gcount());

    bytes_read = count < size ? count : size;

    if (count == 0) {
        if (m_input_file.eof()) {
            return ReturnStatus::success;  // EOF is normal
        }
        if (m_input_file.fail() && !m_input_file.eof()) {
            std::cerr << "Error reading from file" << std::endl;
            return ReturnStatus::failure;
        }
    }

    return ReturnStatus::success;
}

ReturnStatus MediaFileReader::read_and_copy(byte_t* dest_buffer, size_t size, size_t& bytes_read, bool pad_partial)
{
    if (!m_memory_utils) {
        std::cerr << "Memory utils not available" << std::endl;
        return ReturnStatus::failure;
    }

    auto temp_buffer = std::make_unique<byte_t[]>(size);
    auto rc = read_chunk(temp_buffer.get(), size, bytes_read);

    if (rc == ReturnStatus::success && bytes_read > 0) {
        if (bytes_read >= size) {
            m_memory_utils->memory_copy(dest_buffer, temp_buffer.get(), size);
        }
        // Pad remaining bytes with zeros if requested and we got a partial read
        if (pad_partial && bytes_read < size) {
            m_memory_utils->memory_copy(dest_buffer, temp_buffer.get(), bytes_read);
            m_memory_utils->memory_set(dest_buffer + bytes_read, 0, size - bytes_read);
            bytes_read = size;  // Report full size was "read"
        }
    }

    return rc;
}

void MediaFileReader::reset()
{
    if (m_input_file.is_open()) {
        m_input_file.clear();
        m_input_file.seekg(0, std::ios::beg);
    }
}

bool MediaFileReader::handle_eof()
{
    if (m_loop_on_eof && m_input_file.eof()) {
        reset();
        return true;  // Continue reading
    }
    return false;  // Stop reading
}
