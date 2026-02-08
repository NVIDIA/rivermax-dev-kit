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

#ifndef RDK_SERVICES_MEDIA_MEDIA_H_
#define RDK_SERVICES_MEDIA_MEDIA_H_

#include "rdk/services/media/app_media_settings.h"
#include "rdk/services/media/media_defs.h"
#include "rdk/services/media/media_file_reader.h"
#include "rdk/services/media/media_essence_source.h"
#include "rdk/services/media/ancillary_essence_source.h"
#include "rdk/services/media/null_essence_source.h"
#include "rdk/services/media/media_file_essence_source.h"
#include "rdk/services/media/media_file_streaming_provider.h"
#include "rdk/services/media/buffered_essence_source.h"
#include "rdk/services/media/media_essence_sink.h"
#include "rdk/services/media/null_essence_sink.h"
#include "rdk/services/media/media_unit_pool.h"
#include "rdk/services/media/media_settings.h"
#include "rdk/services/media/media_settings_video.h"
#include "rdk/services/media/media_settings_audio.h"
#include "rdk/services/media/media_settings_ancillary.h"
#include "rdk/services/media/media_settings_calculator.h"
#include "rdk/services/media/video_settings_calculator.h"
#include "rdk/services/media/audio_settings_calculator.h"
#include "rdk/services/media/ancillary_settings_calculator.h"
#include "rdk/services/media/afd_encoder.h"
#include "rdk/services/media/closed_caption_608_encoder.h"
#include "rdk/services/media/timecode_encoder.h"
#include "rdk/services/media/closed_caption_source_interface.h"
#include "rdk/services/media/closed_caption_mock_source.h"
#include "rdk/services/media/closed_caption_srt_source.h"

#endif /* RDK_SERVICES_MEDIA_MEDIA_H_ */
