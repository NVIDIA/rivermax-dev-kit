# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

project(rivermax-dev-kit-services-media)

add_library(${PROJECT_NAME} STATIC)

target_sources(${PROJECT_NAME}
    PRIVATE
        ipmx.cpp
        media_file_reader.cpp
        media_file_streaming_provider.cpp
        media_unit_pool.cpp
        afd_encoder.cpp
        timecode_encoder.cpp
        closed_caption_608_encoder.cpp
        ancillary_essence_source.cpp
        closed_caption_mock_source.cpp
        closed_caption_srt_source.cpp
        media_essence_source.cpp
        buffered_essence_source.cpp
        media_file_essence_source.cpp
        null_essence_source.cpp
        null_essence_sink.cpp
        media_settings.cpp
        media_settings_calculator.cpp
        video_settings_calculator.cpp
        audio_settings_calculator.cpp
        ancillary_settings_calculator.cpp
        media_settings_video.cpp
        media_settings_audio.cpp
        media_settings_ancillary.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC include)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        rivermax-dev-kit-build
        rivermax-dev-kit-services-legacy-util
        rivermax-dev-kit-services-ulp-packet
    PUBLIC
        Rivermax::Include
        rivermax-dev-kit-services-error-handling
        rivermax-dev-kit-services-memory-allocation
        rivermax-dev-kit-services-sdp
        rivermax-dev-kit-services-settings
        rivermax-dev-kit-services-utils
)

target_link_libraries(rivermax-dev-kit-services INTERFACE ${PROJECT_NAME})
