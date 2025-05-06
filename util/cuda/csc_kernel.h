/*
 * Copyright © 2021-2025 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its
 * affiliates (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
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

