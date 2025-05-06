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

#ifndef RDK_SERVICES_MEDIA_MEDIA_FRAME_PROVIDER_H_
#define RDK_SERVICES_MEDIA_MEDIA_FRAME_PROVIDER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <chrono>
#include <unordered_map>
#include <functional>

#include "rdk/services/utils/defs.h"
#include "rdk/services/memory_allocation/memory_allocation.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{
/**
 *
 * @brief: Holds the essential frame data.
 */
class FrameBuffer {
private:
    /* Smart pointer for the owned memory */
    std::unique_ptr<byte_t[]> owned_buffer;
    /* Raw pointer that always points to the active buffer (whether owned or borrowed) */
    byte_t* buffer_ptr;
    size_t size;
    bool is_owned;
public:
    /**
     * @brief: Constructor that allocates internal memory.
     *
     * @param [in] buffer_size: Size of the buffer to allocate.
     */
    FrameBuffer(size_t buffer_size);
    /**
     * @brief: Constructor for external memory given as a raw pointer.
     *
     * @param [in] external_buffer: Pointer to the external buffer.
     * @param [in] buffer_size: Size of the external buffer.
     */
    FrameBuffer(byte_t* external_buffer, size_t buffer_size);
    /**
     * @brief: Constructor for external memory provided as a shared_ptr.
     *
     * @param [in] shared_buffer: Shared pointer to the external buffer.
     * @param [in] buffer_size: Size of the external buffer.
     */
    FrameBuffer(const std::shared_ptr<byte_t>& shared_buffer, size_t buffer_size);
    FrameBuffer(const FrameBuffer&) = delete;
    FrameBuffer& operator=(const FrameBuffer&) = delete;
    FrameBuffer(FrameBuffer&& other) noexcept;
    FrameBuffer& operator=(FrameBuffer&& other) noexcept;
    ~FrameBuffer() = default;
    /**
     * @brief: Return pointer to the buffer.
     *
     * @return: Pointer to the buffer.
     */
    byte_t* get() const {
        return buffer_ptr;
    }
    /**
     * @brief: Return size of the buffer.
     *
     * @return: Size of the buffer.
     */
    size_t get_size() const {
        return size;
    }
};

/**
 * @brief: Holds metadata for a frame.
 */
struct FrameMetadata {
    MediaType media_type;
    Resolution resolution;
    uint32_t sequence_number = 0;
    /* Additional metadata */
    std::unordered_map<std::string, std::string> additional_info;
};

/**
 * @brief: Represents a media frame with data and metadata.
 */
struct MediaFrame {
    FrameBuffer data;
    std::shared_ptr<FrameMetadata> metadata;
    /**
     * @brief: Constructor that allocates its own memory.
     *
     * @param [in] buffer_size: Size of the buffer to allocate.
     */
    MediaFrame(size_t buffer_size);
    /**
     * @brief: Constructor for external memory provided as a raw pointer.
     *
     * @param [in] external_buffer: Pointer to the external buffer.
     * @param [in] buffer_size: Size of the external buffer.
     */
    MediaFrame(byte_t* external_buffer, size_t buffer_size);
    /**
     * @brief: Constructor for external memory provided as a shared_ptr.
     *
     * @param [in] shared_buffer: Shared pointer to the external buffer.
     * @param [in] buffer_size: Size of the external buffer.
     */
    MediaFrame(const std::shared_ptr<byte_t>& shared_buffer, size_t buffer_size);
    MediaFrame(const MediaFrame&) = delete;
    MediaFrame& operator=(const MediaFrame&) = delete;
    MediaFrame(MediaFrame&&) noexcept = default;
    MediaFrame& operator=(MediaFrame&&) noexcept = default;
    /**
     * @brief: Adds metadata to the frame.
     *
     * @param [in] metadata_: Metadata to add.
     */
    void add_metadata(const FrameMetadata &metadata_)  { metadata = std::make_shared<FrameMetadata>(metadata_); }
};

/**
 * @brief: Interface for frame providers.
 */
class IFrameProvider {
public:
    virtual ~IFrameProvider() = default;
    /**
     * @brief: Returns a frame in a blocking manner.
     *
     * @return: Shared pointer to the media frame.
     */
    virtual std::shared_ptr<MediaFrame> get_frame_blocking() = 0;
    /**
     * @brief: Returns a frame in a non-blocking manner.
     *
     * @return: Shared pointer to the media frame.
     */
    virtual std::shared_ptr<MediaFrame> get_frame_not_blocking() = 0;
    /**
     * @brief: Stop the provider and release all waiting threads.
     */
    virtual void stop() {};
protected:
    IFrameProvider() = default;
};

/**
 * @brief: Mock implementation of IFrameProvider for testing.
 */
class NullFrameProvider : public IFrameProvider {
protected:
    MediaSettings m_media_settings;
    FrameMetadata m_cached_metadata;
    size_t m_cached_frame_size;
    float m_frame_not_available_probability;
public:
    /**
     * @brief: Constructor of NullFrameProvider.
     *
     * @param [in] media_settings: Media settings for the generated stream.
     */
    NullFrameProvider(const MediaSettings& media_settings);

    std::shared_ptr<MediaFrame> get_frame_blocking() override;
    std::shared_ptr<MediaFrame> get_frame_not_blocking() override;
    /**
     * @brief: Sets the probability of returning nullptr in @ref get_frame_not_blocking.
     *
     * @param [in] probability: Probability value between 0 and 1.
     *
     * @return: Status of the operation.
     */
    ReturnStatus set_frame_not_available_probability(float probability);
    /**
     * @brief: Returns the current probability of returning nullptr in @ref get_frame_not_blocking.
     *
     * @return: Probability value between 0 and 1.
     */
    float get_frame_not_available_probability() const  { return m_frame_not_available_probability; }
private:
    /**
     * @brief: Sets frame settings based on the stream type.
     *
     * @param [in] metadata: Metadata to set.
     * @param [in] frame_size: Size of the frame to set.
     */
    void set_frame_settings(FrameMetadata& metadata, size_t& frame_size);
    /**
     * @brief: Determines if a frame should be available based on the probability.
     *
     * @return: True if the frame should be available, false otherwise.
     */
    bool is_frame_available() const;
};

/**
 * @brief: Provide media frames from a buffer queue, implementing @ref IFrameProvider.
 *
 * This class maintains a queue of media frames and provides them to consumers.
 * Frames can be added to the queue by producers and will be automatically
 * returned to their source when no longer needed.
 */
class BufferedMediaFrameProvider : public IFrameProvider {
private:
    /* Queue of media frames */
    std::queue<std::shared_ptr<MediaFrame>> m_frame_queue;
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
     * @param [in] max_queue_size: Maximum size of the internal frame queue (0 for unlimited).
     */
    BufferedMediaFrameProvider(size_t max_queue_size = 0);
    /**
     * @brief: Destructor.
     */
    virtual ~BufferedMediaFrameProvider();

    std::shared_ptr<MediaFrame> get_frame_blocking() override;
    std::shared_ptr<MediaFrame> get_frame_not_blocking() override;
    void stop() override;
    /**
     * @brief: Add a frame to the queue.
     *
     * @param [in] frame: Shared pointer to a @ref MediaFrame.
     *
     * @return: Status of the operation.
     */
    virtual ReturnStatus add_frame(std::shared_ptr<MediaFrame> frame);
    /**
     * @brief: Return the number of frames in the queue.
     *
     * @return: Number of frames in the queue.
     */
    size_t get_queue_size() const;
};

/**
 * @brief: Reads frames from a binary file.
 *
 * It loads frames into a queue and, if the loop option is enabled,
 * re-inserts frames after serving them.
 */
class MediaFileFrameProvider : public IFrameProvider {
protected:
    std::string m_file_path;
    MemoryAllocator& m_mem_allocator;
    MediaType m_media_type;
    size_t m_frame_size;
    bool m_loop_frames;
    bool m_stop;
    bool m_frames_loaded;
    std::ifstream m_input_file;
    std::queue<std::shared_ptr<MediaFrame>> m_frame_queue;
    std::mutex m_mutex;
    std::condition_variable m_cv;
public:
    /**
     * @brief: Constructor.
     *
     * @param [in] file_path: Path to the media file.
     * @param [in] type: Type of stream.
     * @param [in] frame_size: Size of each frame.
     * @param [in] mem_allocator: Memory allocator to use.
     * @param [in] loop: Whether to loop frames.
     */
    MediaFileFrameProvider(const std::string &file_path, MediaType type, size_t frame_size,
        MemoryAllocator& mem_allocator, bool loop = false);
    /**
     * @brief: Destructor.
     */
    ~MediaFileFrameProvider();

    std::shared_ptr<MediaFrame> get_frame_blocking() override;
    std::shared_ptr<MediaFrame> get_frame_not_blocking() override;
    void stop() override;
    /**
     * @brief: Loads frames from the file.
     */
    ReturnStatus load_frames();
private:
    /**
     * @brief: Handles looping of frames.
     *
     * @param [in] frame: Frame to handle.
     */
    void handle_looping_frame(std::shared_ptr<MediaFrame>& frame);
    /**
     * @brief: Allocates memory for the frames.
     *
     * @param [in] file_size: Size of the file.
     * @param [out] file_memory_buffer: Pointer to the allocated memory.
     * @param [out] required_memory_size: Size of the allocated memory.
     *
     * @return: Status of the operation.
     */
    ReturnStatus allocate_frames_memory(size_t file_size,
        byte_t*& file_memory_buffer, size_t& required_memory_size);
    /**
     * @brief: Reads frames from the file and put them in the queue.
     *
     * @param [in] file_memory_buffer: Pointer to the memory buffer.
     *
     * @return: Status of the operation.
     */
    ReturnStatus read_frames(byte_t* file_memory_buffer);
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_MEDIA_FRAME_PROVIDER_H_ */
