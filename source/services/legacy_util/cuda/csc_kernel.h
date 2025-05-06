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

#include <inc/helper_cuda.h>

// CUDA device constants
__device__ __constant__ float KB;
__device__ __constant__ float KR;
__device__ __constant__ float A1;
__device__ __constant__ float A2;
__device__ __constant__ float InvA1;
__device__ __constant__ float InvA2;

// CUDA Processing Functions
extern "C" void YCrCbToRGB_10bit(uint8_t *src, ushort4 *dst, uint imageW, uint imageH);
extern "C" void YCrCbToRGB_8bit(uint8_t *src, uchar4 *dst, uint imageW, uint imageH);
extern "C" void RGBToRGBA_8bit(uint8_t * src, uchar4 * dst, uint imageW, uint imageH);

