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

#ifndef RDK_SERVICES_MEDIA_MEDIA_ESSENCE_PROVIDER_H_
#define RDK_SERVICES_MEDIA_MEDIA_ESSENCE_PROVIDER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <unordered_map>

#include "rdk/services/utils/defs.h"
#include "rdk/services/media/media_settings.h"
#include "rdk/services/memory_allocation/memory_allocation.h"

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
struct MediaUnitMetadata {
    SMPTEStandard smpte_standard;
    Resolution resolution;
    uint32_t sequence_number = 0;
    /* Additional metadata */
    std::unordered_map<std::string, std::string> additional_info;
};

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
     */
    MediaUnit(size_t buffer_size);
    /**
     * @brief: Constructor for external memory provided as a raw pointer.
     *
     * @param [in] external_buffer: Pointer to the external buffer.
     * @param [in] buffer_size: Size of the external buffer.
     * @param [in] memory_location: Memory location of the external buffer.
     */
    MediaUnit(byte_t* external_buffer, size_t buffer_size,
        MemoryLocation memory_location = MemoryLocation::Host);
    /**
     * @brief: Constructor for external memory provided as a shared_ptr.
     *
     * @param [in] shared_buffer: Shared pointer to the external buffer.
     * @param [in] buffer_size: Size of the external buffer.
     * @param [in] memory_location: Memory location of the external buffer.
     */
    MediaUnit(const std::shared_ptr<byte_t>& shared_buffer, size_t buffer_size,
        MemoryLocation memory_location = MemoryLocation::Host);
    /**
     * @brief: Constructor for external memory provided as a unique_ptr.
     *
     * @param [in] unit_data: Unique pointer to an @ref IMediaUnitBuffer implementation.
     */
    MediaUnit(std::unique_ptr<IMediaUnitBuffer>&& unit_data);
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
};

/**
 * @brief: Interface for media essence providers.
 */
class IMediaEssenceProvider {
public:
    virtual ~IMediaEssenceProvider() = default;
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
     * @brief: Stop the provider and release all waiting threads.
     */
    virtual void stop() {};
protected:
    IMediaEssenceProvider() = default;
};

/**
 * @brief: Mock implementation of IMediaEssenceProvider for testing.
 */
class NullEssenceProvider : public IMediaEssenceProvider {
protected:
    const MediaSettings& m_media_settings;
    MediaUnitMetadata m_cached_metadata;
    size_t m_cached_media_unit_size;
    float m_media_unit_not_available_probability;
public:
    /**
     * @brief: Constructor of NullEssenceProvider.
     *
     * @param [in] media_settings: Media settings for the generated stream.
     */
    NullEssenceProvider(const MediaSettings& media_settings);

    std::shared_ptr<MediaUnit> get_media_unit_blocking() override;
    std::shared_ptr<MediaUnit> get_media_unit_non_blocking() override;
    /**
     * @brief: Sets the probability of returning nullptr in @ref get_media_unit_non_blocking.
     *
     * @param [in] probability: Probability value between 0 and 1.
     *
     * @return: Status of the operation.
     */
    ReturnStatus set_media_unit_not_available_probability(float probability);
    /**
     * @brief: Returns the current probability of returning nullptr in @ref get_media_unit_non_blocking.
     *
     * @return: Probability value between 0 and 1.
     */
    float get_media_unit_not_available_probability() const  { return m_media_unit_not_available_probability; }
private:
    /**
     * @brief: Sets media unit settings based on the stream type.
     *
     * @param [in] metadata: Metadata to set.
     * @param [in] media_unit_size: Size of the media unit to set.
     */
    void set_media_unit_settings(MediaUnitMetadata& metadata, size_t& media_unit_size);
    /**
     * @brief: Determines if a media unit should be available based on the probability.
     *
     * @return: True if the media unit should be available, false otherwise.
     */
    bool is_media_unit_available() const;
};

/**
 * @brief: Provide media units from a buffer queue, implementing @ref IMediaEssenceProvider.
 *
 * This class maintains a queue of media units and provides them to consumers.
 * Units can be added to the queue by producers and will be automatically
 * returned to their source when no longer needed.
 */
class BufferedEssenceProvider : public IMediaEssenceProvider {
private:
    /* Queue of media units */
    std::queue<std::shared_ptr<MediaUnit>> m_media_unit_queue;
    /* Mutex for thread safety */
    mutable std::mutex m_mutex;
    /* Condition variable for blocking operations */
    std::condition_variable m_cv;
    /* Maximum size of the queue (0 for unlimited) */
    size_t m_max_queue_size;
    /* Flag to indicate if the provider is stopping */
    bool m_stop;
public:
    /**
     * @brief: Constructor.
     *
     * @param [in] max_queue_size: Maximum size of the internal media unit queue (0 for unlimited).
     */
    BufferedEssenceProvider(size_t max_queue_size = 0);
    /**
     * @brief: Destructor.
     */
    virtual ~BufferedEssenceProvider();

    std::shared_ptr<MediaUnit> get_media_unit_blocking() override;
    std::shared_ptr<MediaUnit> get_media_unit_non_blocking() override;
    void stop() override;
    /**
     * @brief: Add a media unit to the queue.
     *
     * @param [in] media_unit: Shared pointer to a @ref MediaUnit.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus add_media_unit(std::shared_ptr<MediaUnit> media_unit);
    /**
     * @brief: Return the number of media units in the queue.
     *
     * @return: Number of media units in the queue.
     */
    size_t get_queue_size() const;
};

/**
 * @brief: Reads media units from a binary file.
 *
 * It loads media units into a queue and, if the loop option is enabled,
 * re-inserts media units after serving them.
 */
class MediaFileEssenceProvider : public IMediaEssenceProvider {
protected:
    std::string m_file_path;
    MemoryAllocator& m_mem_allocator;
    SMPTEStandard m_smpte_standard;
    size_t m_media_unit_size;
    size_t m_aligned_media_unit_size;
    bool m_loop_media_units;
    bool m_stop;
    bool m_media_units_loaded;
    std::ifstream m_input_file;
    std::queue<std::shared_ptr<MediaUnit>> m_media_unit_queue;
    std::mutex m_mutex;
    std::condition_variable m_cv;
public:
    /**
     * @brief: Constructor.
     *
     * @param [in] file_path: Path to the media file.
     * @param [in] smpte_standard: SMPTE standard.
     * @param [in] media_unit_size: Size of each media unit.
     * @param [in] mem_allocator: Memory allocator to use.
     * @param [in] loop: Whether to loop media units.
     */
    MediaFileEssenceProvider(const std::string &file_path, SMPTEStandard smpte_standard, size_t media_unit_size,
        MemoryAllocator& mem_allocator, bool loop = false);
    /**
     * @brief: Destructor.
     */
    ~MediaFileEssenceProvider();

    std::shared_ptr<MediaUnit> get_media_unit_blocking() override;
    std::shared_ptr<MediaUnit> get_media_unit_non_blocking() override;
    void stop() override;
    /**
     * @brief: Loads media units from the file.
     */
    ReturnStatus load_media_units();
private:
    /**
     * @brief: Handles looping of media units.
     *
     * @param [in] media_unit: Media unit to handle.
     */
    void handle_looping_media_unit(std::shared_ptr<MediaUnit>& media_unit);
    /**
     * @brief: Allocates memory for the media units.
     *
     * @param [in] file_size: Size of the file.
     * @param [out] file_memory_buffer: Pointer to the allocated memory.
     * @param [out] required_memory_size: Size of the allocated memory.
     *
     * @return: Status of the operation.
     */
    ReturnStatus allocate_media_units_memory(size_t file_size,
        byte_t*& file_memory_buffer, size_t& required_memory_size);
    /**
     * @brief: Reads media units from the file and puts them in the queue.
     *
     * @param [in] file_memory_buffer: Pointer to the memory buffer.
     *
     * @return: Status of the operation.
     */
    ReturnStatus read_media_units(byte_t* file_memory_buffer);
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_MEDIA_ESSENCE_PROVIDER_H_ */
