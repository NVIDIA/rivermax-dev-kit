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

#ifndef RDK_SERVICES_MEDIA_MEDIA_ESSENCE_SOURCE_H_
#define RDK_SERVICES_MEDIA_MEDIA_ESSENCE_SOURCE_H_

#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>

#include "rdk/services/media/media_settings.h"
#include "rdk/services/memory_allocation/memory_allocation.h"
#include "rdk/services/utils/defs.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 * @brief: Abstract interface for media unit buffer management.
 *
 * IMediaUnitBuffer defines the essential contract for media unit buffer implementations,
 * providing access to buffer data, size information, and memory location details.
 * This interface abstracts the underlying memory management strategy, allowing
 * different implementations to handle owned vs. borrowed memory, different
 * memory locations (Host/GPU), and various allocation schemes.
 *
 * Key responsibilities:
 * - Provide access to the raw buffer data.
 * - Report buffer size and alignment information.
 * - Indicate memory location for proper memory operations.
 */
class IMediaUnitBuffer {
public:
    /**
     * @brief: Destructor.
     */
    virtual ~IMediaUnitBuffer() = default;
    /**
     * @brief: Returns pointer to the buffer.
     *
     * @return: Pointer to the buffer.
     */
    virtual byte_t* get() const = 0;
    /**
     * @brief: Returns size of the buffer.
     *
     * @return: Size of the buffer.
     */
    virtual size_t get_size() const = 0;
    /**
     * @brief: Returns aligned size of the buffer.
     *
     * @return: Aligned size of the buffer.
     */
    virtual size_t get_aligned_size() const = 0;
    /**
     * @brief: Returns the memory location of the buffer.
     *
     * @return: Memory location of the buffer.
     **/
    virtual MemoryLocation get_memory_location() const = 0;
};

/**
 * @brief: Concrete implementation of @ref IMediaUnitBuffer with flexible memory management.
 *
 * MediaUnitBuffer provides a robust media unit buffer implementation that supports multiple
 * memory ownership models:
 *
 * 1. Owned Memory: Automatically allocates and manages its own memory buffer.
 * 2. Borrowed Raw Pointer: References externally managed memory via raw pointer.
 * 3. Shared Memory: References externally managed memory via shared_ptr.
 *
 * Key features:
 * - Memory Location Awareness: Tracks whether data resides on Host or GPU.
 * - Move Semantics: Efficient transfer of ownership without copying data.
 * - Zero-Copy Operations: Can reference external buffers without duplication.
 * - RAII Compliance: Automatic resource management with proper cleanup.
 * - Thread Safety: Safe for concurrent read access.
 */
class MediaUnitBuffer : public IMediaUnitBuffer {
private:
    /* Smart pointer for the owned memory */
    std::unique_ptr<byte_t[]> m_owned_buffer;
    /* Raw pointer that always points to the active buffer (whether owned or borrowed) */
    byte_t* m_buffer_ptr;
    size_t m_size;
    bool m_is_owned;
    MemoryLocation m_memory_location;
public:
    /**
     * @brief: Constructor that allocates internal memory.
     *
     * @param [in] buffer_size: Size of the buffer to allocate.
     */
    MediaUnitBuffer(size_t buffer_size);
    /**
     * @brief: Constructor for external memory given as a raw pointer.
     *
     * @param [in] external_buffer: Pointer to the external buffer.
     * @param [in] buffer_size: Size of the external buffer.
     * @param [in] memory_location: Memory location of the external buffer.
     */
    MediaUnitBuffer(byte_t* external_buffer, size_t buffer_size,
        MemoryLocation memory_location = MemoryLocation::Host);
    /**
     * @brief: Constructor for external memory provided as a shared_ptr.
     *
     * @param [in] shared_buffer: Shared pointer to the external buffer.
     * @param [in] buffer_size: Size of the external buffer.
     * @param [in] memory_location: Memory location of the external buffer.
     */
    MediaUnitBuffer(const std::shared_ptr<byte_t>& shared_buffer, size_t buffer_size,
        MemoryLocation memory_location = MemoryLocation::Host);
    MediaUnitBuffer(const MediaUnitBuffer&) = delete;
    MediaUnitBuffer& operator=(const MediaUnitBuffer&) = delete;
    MediaUnitBuffer(MediaUnitBuffer&& other) noexcept;
    MediaUnitBuffer& operator=(MediaUnitBuffer&& other) noexcept;
    ~MediaUnitBuffer() = default;
    byte_t* get() const override { return m_buffer_ptr; }
    size_t get_size() const override { return m_size; }
    size_t get_aligned_size() const override { return m_size; }
    MemoryLocation get_memory_location() const override { return m_memory_location; }
};

/**
 * @brief: Holds metadata for a media unit.
 */
struct MediaUnitMetadata
{
    SMPTEStandard smpte_standard;
    /* Additional metadata */
    std::unordered_map<std::string, std::string> additional_info;

    explicit MediaUnitMetadata(SMPTEStandard _smpte_standard) :
        smpte_standard(_smpte_standard)
    {
    }
    virtual ~MediaUnitMetadata() = default;
};

/**
 * @brief: Callback type for custom metadata factory.
 *
 * Allows users to provide custom metadata creation logic.
 * The callback receives the SMPTE standard and should return a
 * shared_ptr to the appropriate metadata type, or nullptr to
 * use default ones.
 */
using MetadataFactoryCallback = std::function<std::shared_ptr<MediaUnitMetadata>(SMPTEStandard)>;

/**
 * @brief: Represents a media unit with data and metadata.
 *
 * MediaUnit encapsulates media unit data through a polymorphic @ref IMediaUnitBuffer interface
 * and associated metadata. It supports multiple construction patterns for different
 * memory management scenarios:
 *
 * 1. Self-Allocated: Creates its own MediaUnitBuffer with allocated memory
 * 2. External Raw Pointer: Wraps external memory via raw pointer
 * 3. External Shared Pointer: Wraps external memory via shared_ptr
 * 4. Polymorphic Buffer: Accepts any @ref IMediaUnitBuffer implementation
 *
 * The media unit data is managed through a unique_ptr<IMediaUnitBuffer>, enabling
 * polymorphic behavior while maintaining clear ownership semantics.
 * Move-only semantics prevent accidental expensive media unit copies.
 */
struct MediaUnit {
    std::unique_ptr<IMediaUnitBuffer> data;
    std::shared_ptr<MediaUnitMetadata> metadata;
    /**
     * @brief: Constructor that allocates its own memory.
     *
     * @param [in] buffer_size: Size of the buffer to allocate.
     * @param [in] smpte_standard: SMPTE standard to determine metadata type.
     * @param [in] custom_factory: Optional custom metadata factory callback.
     */
    MediaUnit(size_t buffer_size, SMPTEStandard smpte_standard, MetadataFactoryCallback custom_factory = nullptr);
    /**
     * @brief: Constructor for external raw pointer.
     *
     * @param [in] external_buffer: Pointer to the external buffer.
     * @param [in] buffer_size: Size of the external buffer.
     * @param [in] smpte_standard: SMPTE standard to determine metadata type.
     * @param [in] memory_location: Memory location of the external buffer.
     * @param [in] custom_factory: Optional custom metadata factory callback.
     */
    MediaUnit(byte_t* external_buffer,  size_t buffer_size, SMPTEStandard smpte_standard,
        MemoryLocation memory_location = MemoryLocation::Host, MetadataFactoryCallback custom_factory = nullptr);
    /**
     * @brief: Constructor for external shared_ptr.
     *
     * @param [in] shared_buffer: Shared pointer to the external buffer.
     * @param [in] buffer_size: Size of the external buffer.
     * @param [in] smpte_standard: SMPTE standard to determine metadata type.
     * @param [in] memory_location: Memory location of the external buffer.
     * @param [in] custom_factory: Optional custom metadata factory callback.
     */
    MediaUnit(const std::shared_ptr<byte_t>& shared_buffer,  size_t buffer_size, SMPTEStandard smpte_standard,
        MemoryLocation memory_location = MemoryLocation::Host, MetadataFactoryCallback custom_factory = nullptr);
    /**
     * @brief: Constructor for external IMediaUnitBuffer.
     *
     * @param [in] unit_buffer: Unique pointer to an @ref IMediaUnitBuffer implementation.
     * @param [in] smpte_standard: SMPTE standard to determine metadata type.
     * @param [in] custom_factory: Optional custom metadata factory callback.
     */
    MediaUnit(std::unique_ptr<IMediaUnitBuffer>&& unit_buffer, SMPTEStandard smpte_standard,
        MetadataFactoryCallback custom_factory = nullptr);
    MediaUnit(const MediaUnit&) = delete;
    MediaUnit& operator=(const MediaUnit&) = delete;
    MediaUnit(MediaUnit&&) noexcept = default;
    MediaUnit& operator=(MediaUnit&&) noexcept = default;
    /**
     * @brief: Adds metadata to the media unit.
     *
     * @param [in] metadata_: Metadata to add.
     */
    void add_metadata(const MediaUnitMetadata& metadata_)  { metadata = std::make_shared<MediaUnitMetadata>(metadata_); }

private:
    /**
     * @brief: Helper function to create metadata based on SMPTE standard.
     *
     * @param [in] smpte_standard: SMPTE standard.
     * @param [in] custom_factory: Optional custom factory.
     *
     * @return: Shared pointer to created metadata.
     */
    static std::shared_ptr<MediaUnitMetadata> create_metadata(
        SMPTEStandard smpte_standard,
        MetadataFactoryCallback custom_factory);
};

/**
 * @brief: Interface for media essence sources.
 */
class IMediaEssenceSource {
public:
    virtual ~IMediaEssenceSource() = default;
    /**
     * @brief: Returns a media unit in a blocking manner.
     *
     * @return: Shared pointer to the media unit.
     */
    virtual std::shared_ptr<MediaUnit> get_media_unit_blocking() = 0;
    /**
     * @brief: Returns a media unit in a non-blocking manner.
     *
     * @return: Shared pointer to the media unit.
     */
    virtual std::shared_ptr<MediaUnit> get_media_unit_non_blocking() = 0;
    /**
     * @brief: Stop the source and release all waiting threads.
     */
     virtual void stop() {};
    /**
     * @brief: Sets the start time for the source.
     *
     * @param [in] time_ns: The start time.
     */
    virtual void set_start_time(uint64_t time_ns) {};
    /**
     * @brief: Returns the total size of the essence source.
     *
     * @return: Total size of the essence source.
     */
    virtual ReturnStatus get_data_size(size_t& data_size) const {
        data_size = 0;
        return ReturnStatus::success;
    };
protected:
    IMediaEssenceSource() = default;
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_MEDIA_ESSENCE_SOURCE_H_ */
