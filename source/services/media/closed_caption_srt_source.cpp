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

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include "rdk/services/media/closed_caption_srt_source.h"
#include "rdk/services/media/media_defs.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

ClosedCaptionSRTSource::ClosedCaptionSRTSource(const std::string& srt_file_path) :
    m_last_index(0)
{
    if (!parse_srt_file(srt_file_path)) {
        std::cerr << "Failed to parse SRT file: " << srt_file_path << std::endl;
    } else {
        std::cout << "Loaded " << m_subtitles.size() << " subtitle entries from " << srt_file_path << std::endl;
    }
}

std::string ClosedCaptionSRTSource::get_caption_text(uint64_t timestamp_ns)
{
    // Optimize for sequential access: Start from last found index and search forward, then wrap around if needed
    for (size_t index = m_last_index; index < m_subtitles.size(); ++index) {
        const auto& entry = m_subtitles[index];

        if (timestamp_ns >= entry.start_ns && timestamp_ns < entry.end_ns) {
            m_last_index = index;
            return entry.text;
        }
        if (timestamp_ns < entry.start_ns) {
            break;
        }
    }
    for (size_t index = 0; index < m_last_index; ++index) {
        const auto& entry = m_subtitles[index];

        if (timestamp_ns >= entry.start_ns && timestamp_ns < entry.end_ns) {
            m_last_index = index;
            return entry.text;
        }
    }

    return "";
}

bool ClosedCaptionSRTSource::parse_srt_file(const std::string& file_path)
{
    /* SRT file format:
     *   1
     *   HH:MM:SS,mmm --> HH:MM:SS,mmm
     *   Subtitle text line 1
     *   Subtitle text line 2
     *
     *   2
     *   HH:MM:SS,mmm --> HH:MM:SS,mmm
     *   Next subtitle text
     */
    
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line == "\r") {
            continue;
        }

        // First line should be subtitle number
        std::istringstream iss(line);
        int subtitle_number;
        if (!(iss >> subtitle_number)) {
            continue;
        }

        // Second line should be timestamp range
        if (!std::getline(file, line)) {
            break;
        }

        // Parse timestamp line: "HH:MM:SS,mmm --> HH:MM:SS,mmm"
        size_t arrow_pos = line.find("-->");
        if (arrow_pos == std::string::npos) {
            continue;
        }

        // Parse timestamp range and trim whitespace
        std::string start_str = line.substr(0, arrow_pos);
        std::string end_str = line.substr(arrow_pos + 3);
        start_str.erase(0, start_str.find_first_not_of(" \t\r\n"));
        start_str.erase(start_str.find_last_not_of(" \t\r\n") + 1);
        end_str.erase(0, end_str.find_first_not_of(" \t\r\n"));
        end_str.erase(end_str.find_last_not_of(" \t\r\n") + 1);

        uint64_t start_ns = parse_srt_timestamp(start_str);
        uint64_t end_ns = parse_srt_timestamp(end_str);

        // Read subtitle text and concatenate multiple lines
        std::string text;
        while (std::getline(file, line) && !line.empty() && line != "\r") {
            if (!text.empty()) {
                text += " ";
            }
            if (line.back() == '\r') {
                line.pop_back();
            }
            text += line;
        }

        // Make sure timestamp range is valid
        if (!text.empty()) {
            if (end_ns <= start_ns) {
                std::cerr << "Warning: Skipping invalid SRT entry with invalid timestamp range ("
                          << start_str << " --> " << end_str << ")" << std::endl;
            } else {
                m_subtitles.push_back({ start_ns, end_ns, std::move(text) });
            }
        }
    }

    // Sort subtitles by start time to make sure captions are in chronological order
    std::sort(m_subtitles.begin(), m_subtitles.end(),
              [](const SubtitleEntry& a, const SubtitleEntry& b) {
                  return a.start_ns < b.start_ns;
              });

    return !m_subtitles.empty();
}

uint64_t ClosedCaptionSRTSource::parse_srt_timestamp(const std::string& timestamp_str)
{
    // SRT timestamp format: HH:MM:SS,mmm
    int hours = 0;
    int minutes = 0;
    int seconds = 0;
    int milliseconds = 0;

    if (sscanf(timestamp_str.c_str(), "%d:%d:%d,%d", &hours, &minutes, &seconds, &milliseconds) == 4) {
        uint64_t timestamp_ns = 0;
        timestamp_ns += static_cast<uint64_t>(hours) * 3600 * NS_IN_SEC;
        timestamp_ns += static_cast<uint64_t>(minutes) * 60 * NS_IN_SEC;
        timestamp_ns += static_cast<uint64_t>(seconds) * NS_IN_SEC;
        timestamp_ns += static_cast<uint64_t>(milliseconds) * NS_IN_MSEC;
        return timestamp_ns;
    }

    return 0;
}

} // namespace services
} // namespace dev_kit
} // namespace rivermax
