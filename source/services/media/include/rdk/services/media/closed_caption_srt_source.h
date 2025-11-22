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

#ifndef RDK_SERVICES_MEDIA_CLOSED_CAPTION_SRT_SOURCE_H_
#define RDK_SERVICES_MEDIA_CLOSED_CAPTION_SRT_SOURCE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "rdk/services/media/closed_caption_source_interface.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: SRT subtitle file closed caption source.
 *
 * Parses captions strings from SRT subtitle files
 * and builds a list of subtitle entries with start and end timestamps.
 * SRT format:
 *   1
 *   HH:MM:SS,mmm --> HH:MM:SS,mmm
 *   Subtitle text line 1
 *   Subtitle text line 2
 *
 *   2
 *   HH:MM:SS,mmm --> HH:MM:SS,mmm
 *   Next subtitle text
 */
class ClosedCaptionSRTSource : public IClosedCaptionSource
{
public:
    /**
     * @brief: Constructs SRT source from file.
     *
     * @param [in] srt_file_path: Path to .srt subtitle file.
     */
    explicit ClosedCaptionSRTSource(const std::string& srt_file_path);
    std::string get_caption_text(uint64_t timestamp_ns) override;

private:
    /**
     * @brief: Subtitle entry.
     */
    struct SubtitleEntry
    {
        uint64_t start_ns;
        uint64_t end_ns;
        std::string text;
    };
    std::vector<SubtitleEntry> m_subtitles;
    size_t m_last_index;

    /**
     * @brief: Parses SRT file and builds subtitle entries.
     *
     * @param [in] file_path: Path to .srt subtitle file.
     *
     * @return: True if successful, false otherwise.
     */
    bool parse_srt_file(const std::string& file_path);
    /**
     * @brief: Parses SRT timestamp string and converts to nanoseconds.
     *
     * @param [in] timestamp_str: SRT timestamp string.
     *
     * @return: Timestamp in nanoseconds.
     */
    uint64_t parse_srt_timestamp(const std::string& timestamp_str);
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_MEDIA_CLOSED_CAPTION_SRT_SOURCE_H_ */
