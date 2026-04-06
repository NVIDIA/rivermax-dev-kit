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

#include <cstdlib>
#include <cstring>
#include <random>

#include "rdk/services/media/media_essence_source.h"
#include "rdk/services/media/ancillary_metadata.h"
#include "rdk/services/media/media_settings_video.h"

using namespace rdk::services;

MediaUnitBuffer::MediaUnitBuffer(size_t buffer_size) :
    m_owned_buffer(new byte_t[buffer_size]),
    m_buffer_ptr(m_owned_buffer.get()),
    m_size(buffer_size),
    m_is_owned(true),
    m_memory_location(MemoryLocation::Host)
{
}

MediaUnitBuffer::MediaUnitBuffer(byte_t* external_buffer, size_t buffer_size,
    MemoryLocation memory_location) :
    m_owned_buffer(nullptr),
    m_buffer_ptr(external_buffer),
    m_size(buffer_size),
    m_is_owned(false),
    m_memory_location(memory_location)
{
}

MediaUnitBuffer::MediaUnitBuffer(const std::shared_ptr<byte_t>& shared_buffer, size_t buffer_size,
    MemoryLocation memory_location) :
    m_owned_buffer(nullptr),
    m_buffer_ptr(shared_buffer.get()),
    m_size(buffer_size),
    m_is_owned(false),
    m_memory_location(memory_location)
{
}

MediaUnitBuffer::MediaUnitBuffer(MediaUnitBuffer&& other) noexcept :
    m_owned_buffer(std::move(other.m_owned_buffer)),
    m_buffer_ptr(other.m_buffer_ptr),
    m_size(other.m_size),
    m_is_owned(other.m_is_owned),
    m_memory_location(other.m_memory_location)
{
    // If other owned its buffer, its pointer is now invalid after the move.
    if (other.m_is_owned) {
        other.m_buffer_ptr = nullptr;
    }
}

MediaUnitBuffer& MediaUnitBuffer::operator=(MediaUnitBuffer&& other) noexcept
{
    if (this != &other) {
        m_owned_buffer = std::move(other.m_owned_buffer);
        m_buffer_ptr = other.m_buffer_ptr;
        m_size = other.m_size;
        m_is_owned = other.m_is_owned;
        m_memory_location = other.m_memory_location;

        // If other owned its buffer, its pointer is now invalid after the move
        if (other.m_is_owned) {
            other.m_buffer_ptr = nullptr;
        }
    }
    return *this;
}

MediaUnit::MediaUnit(size_t buffer_size, SMPTEStandard smpte_standard,
    MetadataFactoryCallback custom_factory) :
    data(std::make_unique<MediaUnitBuffer>(buffer_size)),
    metadata(create_metadata(smpte_standard, std::move(custom_factory)))
{
}

MediaUnit::MediaUnit(byte_t* external_buffer, size_t buffer_size, SMPTEStandard smpte_standard,
    MemoryLocation memory_location, MetadataFactoryCallback custom_factory) :
    data(std::make_unique<MediaUnitBuffer>(external_buffer, buffer_size, memory_location)),
    metadata(create_metadata(smpte_standard, std::move(custom_factory)))
{
}

MediaUnit::MediaUnit(const std::shared_ptr<byte_t>& shared_buffer, size_t buffer_size,
    SMPTEStandard smpte_standard, MemoryLocation memory_location,
    MetadataFactoryCallback custom_factory) :
    data(std::make_unique<MediaUnitBuffer>(shared_buffer, buffer_size, memory_location)),
    metadata(create_metadata(smpte_standard, std::move(custom_factory)))
{
}

MediaUnit::MediaUnit(std::unique_ptr<IMediaUnitBuffer>&& unit_buffer, SMPTEStandard smpte_standard,
    MetadataFactoryCallback custom_factory) :
    data(std::move(unit_buffer)),
    metadata(create_metadata(smpte_standard, std::move(custom_factory)))
{
}

std::shared_ptr<MediaUnitMetadata> MediaUnit::create_metadata(
    SMPTEStandard smpte_standard, MetadataFactoryCallback custom_factory)
{
    // Try custom factory first and fallback to default if invalid
    if (custom_factory) {
        auto custom_metadata = custom_factory(smpte_standard);
        if (custom_metadata) {
            return custom_metadata;
        }
    }

    std::shared_ptr<MediaUnitMetadata> metadata_instance;
    switch (smpte_standard) {
        case SMPTEStandard::ST_2110_40:
            metadata_instance = std::make_shared<AncillaryMediaUnitMetadata>();
            break;
        default:
            metadata_instance = std::make_shared<MediaUnitMetadata>(smpte_standard);
            break;
    }
    return metadata_instance;
}
