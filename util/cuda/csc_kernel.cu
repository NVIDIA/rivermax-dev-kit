/*
 * Copyright © 2021-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its
 * affiliates (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */
#ifndef _CSC_KERNEL_CU_
#define _CSC_KERNEL_CU_

#ifdef __linux__
#include <sys/uio.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <inc/helper_cuda.h>
#include <inc/helper_math.h>

#include "csc_server.h"

// Utility macros
#define DIVUP(A,B) ((A)%(B) == 0 ? (A)/(B) : ((A) / (B) + 1))

__global__ void
YCrCb422ToRGB_10bit_kernel(uint8_t *src, ushort4 *dst, uint32_t len)
{
    // srcX is an index into the linear YCrCb input image
    // Each 5 bytes define 2 pixels
    uint srcX = (blockIdx.x * blockDim.x * 5) + (threadIdx.x * 5);

    // dstX is an index into the linear RGB output image
    // Process 2 pixels per invocation
    uint dstX = (blockIdx.x * blockDim.x * 2) + (threadIdx.x * 2);

    const float KB = 0.2123;
    const float KR = 0.0722;
    const float A1 = 876;
    const float A2 = 896;
    const float InvA1 = 1/A1;
    const float InvA2 = 1/A2;

    const uint16_t Mask10bit = 0x03FF;

    if (dstX < len) {
        uint SrcDataPos = srcX;

        uint16_t Cr = ((src[SrcDataPos]<<2)|(src[SrcDataPos+1]>>6)) & Mask10bit;
        uint16_t Y1 = ((src[SrcDataPos+1]<<4)|(src[SrcDataPos+2]>>4)) & Mask10bit;
        uint16_t Cb = ((src[SrcDataPos+2]<<6)|(src[SrcDataPos+3]>>2)) & Mask10bit;
        uint16_t Y2 = ((src[SrcDataPos+3]<<8)|(src[SrcDataPos+4])) & Mask10bit;

        float _Y1 = Y1 * InvA1;
        float B1 = __saturatef(_Y1 + ( InvA2 * ( 2*(1-KB) ) * (Cb-512) ));
        float R1 = __saturatef(_Y1 + ( InvA2 * ( 2*(1-KR) ) * (Cr-512) ));
        float G1 = __saturatef((_Y1 - (KR * R1) - (KB * B1)) / (1 - KR - KB));

        float _Y2 = Y2 * InvA1;
        float B2 = __saturatef(_Y2 + ( InvA2 * ( 2*(1-KB) ) * (Cb-512) ));
        float R2 = __saturatef(_Y2 + ( InvA2 * ( 2*(1-KR) ) * (Cr-512) ));
        float G2 = __saturatef((_Y2 - (KR * R2) - (KB * B2)) / (1 - KR - KB));

        uint RGBDataPos = dstX;

        dst[RGBDataPos].x = __float2uint_rn(B1 * 0x3ff);
        dst[RGBDataPos].y = __float2uint_rn(G1 * 0x3ff);
        dst[RGBDataPos].z = __float2uint_rn(R1 * 0x3ff);

        dst[RGBDataPos+1].x = __float2uint_rn(B2 * 0x3ff);
        dst[RGBDataPos+1].y = __float2uint_rn(G2 * 0x3ff);
        dst[RGBDataPos+1].z = __float2uint_rn(R2 * 0x3ff);

    }
}

extern "C"
void YCrCbToRGB_10bit(uint8_t *src, ushort4 *dst, uint32_t len)
{
#define BLOCK_W 16
#define BLOCK_H 16

    // Calculate block size
    dim3 BlockSz(BLOCK_W, BLOCK_H, 1);

    // Calculate grid size.
    // Processing two pixels per thread
    dim3 GridSz(DIVUP(len, BLOCK_W), 1, 1);

    YCrCb422ToRGB_10bit_kernel<<<GridSz, BlockSz>>>(src, dst, len);

    checkCudaErrors(cudaGetLastError());
}

__global__ void
YCrCb422ToRGB_8bit_kernel(uint8_t *src, uchar4 *dst, uint32_t len)
{
    // srcX is an index into the linear YCrCb input image
    // Each 5 bytes define 2 pixels
    uint srcX = (blockIdx.x * blockDim.x * 5) + (threadIdx.x * 5);

    // dstX is an index into the linear RGB output image
    // Process 2 pixels per invocation
    uint dstX = (blockIdx.x * blockDim.x * 2) + (threadIdx.x * 2);

    const float KB = 0.2123;
    const float KR = 0.0722;
    const float A1 = 876;
    const float A2 = 896;
    const float InvA1 = 1/A1;
    const float InvA2 = 1/A2;

    const uint16_t Mask10bit = 0x03FF;

    if (dstX < len) {
        uint SrcDataPos = srcX;

        uint16_t Cr = ((src[SrcDataPos] << 2) | (src[SrcDataPos + 1] >> 6))& Mask10bit; //962
        uint16_t Y1 = ((src[SrcDataPos + 1] << 4) | (src[SrcDataPos + 2] >> 4))& Mask10bit; // 329
        uint16_t Cb = ((src[SrcDataPos + 2] << 6) | (src[SrcDataPos + 3] >> 2))& Mask10bit; // 361
        uint16_t Y2 = ((src[SrcDataPos + 3] << 8) | (src[SrcDataPos + 4]))& Mask10bit; // 329

        float _Y1 = Y1 * InvA1;
        float B1 = __saturatef(_Y1 + ( InvA2 * ( 2.0f * (1.0f - KB) ) * (Cb - 512.0f) ));
        float R1 = __saturatef(_Y1 + ( InvA2 * ( 2.0f * (1.0f - KR) ) * (Cr - 512.0f) ));
        float G1 = __saturatef((_Y1 - (KR * R1) - (KB * B1)) / (1 - KR - KB));

        float _Y2 = Y2 * InvA1;
        float B2 = __saturatef(_Y2 + ( InvA2 * ( 2.0f * (1.0f - KB) ) * (Cb - 512.0f) ));
        float R2 = __saturatef(_Y2 + ( InvA2 * ( 2.0f * (1.0f - KR) ) * (Cr - 512.0f) ));
        float G2 = __saturatef((_Y2 - (KR * R2) - (KB * B2)) / (1 - KR - KB));

        uint RGBDataPos = dstX;

        dst[RGBDataPos].x = __float2uint_rn(B1 * 0xFF);
        dst[RGBDataPos].y = __float2uint_rn(G1 * 0xFF);
        dst[RGBDataPos].z = __float2uint_rn(R1 * 0xFF);

        dst[RGBDataPos + 1].x = __float2uint_rn(B2 * 0xFF);
        dst[RGBDataPos + 1].y = __float2uint_rn(G2 * 0xFF);
        dst[RGBDataPos + 1].z = __float2uint_rn(R2 * 0xFF);
    }
}

extern "C"
void YCrCbToRGB_8bit(uint8_t *src, uchar4 *dst, uint32_t len)
{
#define BLOCK_W 16
#define BLOCK_H 16

    // Calculate block size
    dim3 BlockSz(BLOCK_W, BLOCK_H, 1);

    // Calculate grid size.
    // Processing two pixels per thread
    dim3 GridSz(DIVUP(len, BLOCK_W * 2), 1, 1);

    YCrCb422ToRGB_8bit_kernel<<<GridSz, BlockSz>>>(src, dst, len);

    checkCudaErrors(cudaGetLastError());
}

__global__ void
RGBToRGBA_8bit_kernel(uchar3* src, uchar4* dst, uint32_t len)
{
    uint srcX = (blockIdx.x * blockDim.x) + (threadIdx.x);

    uint dstX = (blockIdx.x * blockDim.x) + (threadIdx.x);

    dst[dstX].x = src[srcX].x;
    dst[dstX].y = src[srcX].y;
    dst[dstX].z = src[srcX].z;
    dst[dstX].w = 255;

}

extern "C"
void RGBToRGBA_8bit(uint8_t* src, uchar4* dst, uint32_t len)
{
#define BLOCK_W 16
#define BLOCK_H 16

    // Calculate block size
    dim3 BlockSz(BLOCK_W, BLOCK_H, 1);

    // Calculate grid size.
    // Processing two pixels per thread
    dim3 GridSz(DIVUP(len, BLOCK_W), 1, 1);

    RGBToRGBA_8bit_kernel << <GridSz, BlockSz >> > ((uchar3*)src, dst, len);

    checkCudaErrors(cudaGetLastError());
}

__global__ void
RGBToRGBA_10bit_kernel(uint8_t* src, uchar4* dst, uint32_t width, uint32_t height)
{
    // srcX and srcY are indices into the flat src RGB image
    // Each 60 bytes in X defines 16 pixels
    uint32_t srcX = blockIdx.x * (blockDim.x * 60) + (threadIdx.x * 60);
    uint32_t srcY = blockIdx.y * blockDim.y + threadIdx.y;

    // dstX and dstY are indices into the output RGBA image
    // Process 16 pixels per invocation
    uint32_t dstX = blockIdx.x * (blockDim.x * 16) + (threadIdx.x * 16);
    uint32_t dstY = blockIdx.y * blockDim.y + threadIdx.y;

    if ((dstX <= width - 16) && (dstY < height))
    {
        // Each 60 bytes defines 16 pixels
        uint32_t RGBDataPos = srcY * (width * 60 / 16) + srcX;
        uint32_t RGBADataPos = dstY * width + dstX;

        // Unpack 8 MSBs into unsigned bytes dropping 2 LSBs.  Bytes are in MSB order.

        // Pixel 1
        uint32_t tmp = src[RGBDataPos + 0] << 24 | src[RGBDataPos + 1] << 16 | src[RGBDataPos + 2] << 8 | src[RGBDataPos + 3];
        dst[RGBADataPos].x = tmp >> 24 & 0xFF;
        dst[RGBADataPos].y = tmp >> 14 & 0xFF;
        dst[RGBADataPos].z = tmp >> 4 & 0xFF;

        // Pixel 2
        uint32_t extra = tmp << 6;
        tmp = src[RGBDataPos + 4] << 24 | src[RGBDataPos + 5] << 16 | src[RGBDataPos + 6] << 8 | src[RGBDataPos + 7];
        dst[RGBADataPos + 1].x = extra | tmp >> 26 & 0xFF;
        dst[RGBADataPos + 1].y = tmp >> 16 & 0xFF;
        dst[RGBADataPos + 1].z = tmp >> 6 & 0xFF;

        // Pixel 3
        extra = tmp << 4;
        tmp = src[RGBDataPos + 8] << 24 | src[RGBDataPos + 9] << 16 | src[RGBDataPos + 10] << 8 | src[RGBDataPos + 11];
        dst[RGBADataPos + 2].x = extra | tmp >> 28 & 0xFF;
        dst[RGBADataPos + 2].y = tmp >> 18 & 0xFF;
        dst[RGBADataPos + 2].z = tmp >> 8 & 0xFF;

        // Pixel 4 
        extra = tmp << 2;
        tmp = src[RGBDataPos + 12] << 24 | src[RGBDataPos + 13] << 16 | src[RGBDataPos + 14] << 8 | src[RGBDataPos + 15];
        dst[RGBADataPos + 3].x = extra | tmp >> 30 & 0xFF;
        dst[RGBADataPos + 3].y = tmp >> 20 & 0xFF;
        dst[RGBADataPos + 3].z = tmp >> 10 & 0xFF;

        // Pixel 5
        dst[RGBADataPos + 4].x = tmp & 0xFF;
        tmp = src[RGBDataPos + 16] << 24 | src[RGBDataPos + 17] << 16 | src[RGBDataPos + 18] << 8 | src[RGBDataPos + 19];
        dst[RGBADataPos + 4].y = tmp >> 22 & 0xFF;
        dst[RGBADataPos + 4].z = tmp >> 12 & 0xFF;

        // Pixel 6 
        dst[RGBADataPos + 5].x = tmp >> 2 & 0xFF;
        tmp = src[RGBDataPos + 20] << 24 | src[RGBDataPos + 21] << 16 | src[RGBDataPos + 22] << 8 | src[RGBDataPos + 23];
        dst[RGBADataPos + 5].y = tmp >> 24 & 0xFF;
        dst[RGBADataPos + 5].z = tmp >> 14 & 0xFF;

        // Pixel 7
        dst[RGBADataPos + 6].x = tmp >> 4 & 0xFF;
        extra = tmp << 6;
        tmp = src[RGBDataPos + 24] << 24 | src[RGBDataPos + 25] << 16 | src[RGBDataPos + 26] << 8 | src[RGBDataPos + 27];
        dst[RGBADataPos + 6].y = extra | tmp >> 26 & 0xFF;
        dst[RGBADataPos + 6].z = tmp >> 16 & 0xFF;

        // Pixel 8
        dst[RGBADataPos + 7].x = tmp >> 6 & 0xFF;
        extra = tmp << 4;
        tmp = src[RGBDataPos + 28] << 24 | src[RGBDataPos + 29] << 16 | src[RGBDataPos + 30] << 8 | src[RGBDataPos + 31];
        dst[RGBADataPos + 7].y = extra | tmp >> 28 & 0xFF;;
        dst[RGBADataPos + 7].z = tmp >> 18 & 0xFF;;

        // Pixel 9
        dst[RGBADataPos + 8].x = tmp >> 8 & 0xFF;
        extra = tmp << 2;
        tmp = src[RGBDataPos + 32] << 24 | src[RGBDataPos + 33] << 16 | src[RGBDataPos + 34] << 8 | src[RGBDataPos + 35];
        dst[RGBADataPos + 8].y = extra | tmp >> 30 & 0xFF;
        dst[RGBADataPos + 8].z = tmp >> 20 & 0xFF;

        // Pixel 10
        dst[RGBADataPos + 9].x = tmp >> 10 & 0xFF;
        dst[RGBADataPos + 9].y = tmp & 0xFF;
        tmp = src[RGBDataPos + 36] << 24 | src[RGBDataPos + 37] << 16 | src[RGBDataPos + 38] << 8 | src[RGBDataPos + 39];
        dst[RGBADataPos + 9].z = tmp >> 22 & 0xFF;

        // Pixel 11
        dst[RGBADataPos + 10].x = tmp >> 12 & 0xFF;
        dst[RGBADataPos + 10].y = tmp >> 2 & 0xFF;
        tmp = src[RGBDataPos + 40] << 24 | src[RGBDataPos + 41] << 16 | src[RGBDataPos + 42] << 8 | src[RGBDataPos + 43];
        dst[RGBADataPos + 10].z = tmp >> 24 & 0xFF;

        // Pixel 12
        dst[RGBADataPos + 11].x = tmp >> 14 & 0xFF;
        dst[RGBADataPos + 11].y = tmp >> 4 & 0xFF;
        extra = tmp << 6;
        tmp = src[RGBDataPos + 44] << 24 | src[RGBDataPos + 45] << 16 | src[RGBDataPos + 46] << 8 | src[RGBDataPos + 47];
        dst[RGBADataPos + 11].z = extra | tmp >> 26 & 0xFF;

        // Pixel 13
        dst[RGBADataPos + 12].x = tmp >> 16 & 0xFF;
        dst[RGBADataPos + 12].y = tmp >> 6 & 0xFF;
        extra = tmp << 4;
        tmp = src[RGBDataPos + 48] << 24 | src[RGBDataPos + 49] << 16 | src[RGBDataPos + 50] << 8 | src[RGBDataPos + 51];
        dst[RGBADataPos + 12].z = extra | tmp >> 28 & 0xFF;

        // Pixel 14
        dst[RGBADataPos + 13].x = tmp >> 18 & 0xFF;
        dst[RGBADataPos + 13].y = tmp >> 8 & 0xFF;
        extra = tmp << 2;
        tmp = src[RGBDataPos + 52] << 24 | src[RGBDataPos + 53] << 16 | src[RGBDataPos + 54] << 8 | src[RGBDataPos + 55];
        dst[RGBADataPos + 13].z = extra | tmp >> 30 & 0xFF;

        // Pixel 15
        dst[RGBADataPos + 14].x = tmp >> 20 & 0xFF;
        dst[RGBADataPos + 14].y = tmp >> 10 & 0xFF;
        dst[RGBADataPos + 14].z = tmp & 0xFF;

        // Pixel 16
        tmp = src[RGBDataPos + 56] << 24 | src[RGBDataPos + 57] << 16 | src[RGBDataPos + 58] << 8 | src[RGBDataPos + 59];
        dst[RGBADataPos + 15].x = tmp >> 22 & 0xFF;
        dst[RGBADataPos + 15].y = tmp >> 12 & 0xFF;
        dst[RGBADataPos + 15].z = tmp >> 2 & 0xFF;
    }
}

extern "C"
void RGBToRGBA_10bit(uint8_t * src, uchar4 * dst, uint32_t width, uint32_t height)
{
#define BLOCK_W 16
#define BLOCK_H 16

    // Calculate block size
    dim3 BlockSz(BLOCK_W, BLOCK_H, 1);

    // Calculate grid size.
    // Processing sixteen pixels per thread
    dim3 GridSz(DIVUP(width, BLOCK_W), DIVUP(height, BLOCK_H), 1);

    void* args[] = { &src, &dst, &width, &height };
    cudaLaunchKernel((void*)RGBToRGBA_10bit_kernel, GridSz, BlockSz, args, 0);
}

__global__ void
RGBToRGBA_12bit_kernel(uint8_t* src, uchar4* dst, uint32_t width, uint32_t height)
{
    // srcX and srcY are indices into the flat src RGB image
    // Each 72 bytes in X defines 16 pixels
    uint32_t srcX = blockIdx.x * (blockDim.x * 72) + (threadIdx.x * 72);
    uint32_t srcY = blockIdx.y * blockDim.y + threadIdx.y;

    // dstX and dstY are indices into the output RGBA image
    // Process 16 pixels per invocation
    uint32_t dstX = blockIdx.x * (blockDim.x * 16) + (threadIdx.x * 16);
    uint32_t dstY = blockIdx.y * blockDim.y + threadIdx.y;

    if ((dstX <= width - 16) && (dstY < height))
    {
        // Each 72 bytes contain 16 pixels
        uint32_t RGBDataPos = srcY * (width * 72 / 16) + srcX;
        uint32_t RGBADataPos = dstY * width + dstX;

        // Unpack 8 MSBs into unsigned bytes dropping 4 LSBs.  Bytes are in MSB order.

        // 9byte equals 2 pixels, we can use that to iterate      
        #pragma unroll 8
        for (int i = 0; i < 8; ++i)
        {
            uint32_t RGBDataPosI = RGBDataPos + 9 * i;
            uint32_t RGBADataPosI = RGBADataPos + 2 * i;

            //TODO: Perf - load the 4 values as one 32bit int
            uint64_t tmp = src[RGBDataPosI + 0];
            tmp = (src[RGBDataPosI + 1] << 24 | src[RGBDataPosI + 2] << 16 | src[RGBDataPosI + 3] << 8 | src[RGBDataPosI + 4] & 0xFFFFFFFF) | tmp << 32;
            
            dst[RGBADataPosI].x = tmp >> 32 & 0xFF;
            dst[RGBADataPosI].y = tmp >> 20 & 0xFF;
            dst[RGBADataPosI].z = tmp >> 8 & 0xFF;
   
            tmp = ((src[RGBDataPosI + 5] << 24 | src[RGBDataPosI + 6] << 16 | src[RGBDataPosI + 7] << 8 | src[RGBDataPosI + 8]) & 0xFFFFFFFF) | tmp << 32;
            
            dst[RGBADataPosI + 1].x = (tmp >> 28) & 0xFF;
            dst[RGBADataPosI + 1].y = (tmp >> 16) & 0xFF;
            dst[RGBADataPosI + 1].z = (tmp >> 4) & 0xFF;
        }
    }
}

extern "C"
void RGBToRGBA_12bit(uint8_t * src, uchar4 * dst, uint32_t width, uint32_t height)
{
#define BLOCK_W 16
#define BLOCK_H 16

    // Calculate block size
    dim3 BlockSz(BLOCK_W, BLOCK_H, 1);

    // Calculate grid size.
    // Processing sixteen pixels per thread
    dim3 GridSz(DIVUP(width, BLOCK_W), DIVUP(height, BLOCK_H), 1);

    void* args[] = { &src, &dst, &width, &height };
    cudaLaunchKernel((void*)RGBToRGBA_12bit_kernel, GridSz, BlockSz, args, 0);
}


#ifdef __linux__

extern "C"

void YCrCbToRGB_8bit_iovec(iovec *in, uchar4 *dst, uint32_t num)
{
#define BLOCK_W 16
#define BLOCK_H 16

    uint8_t *src;
    uchar4 *dst_temp;
    uint32_t index=0;

    // Calculate block size
    dim3 BlockSz(BLOCK_W, BLOCK_H, 1);

    if (num==1) {
        // Frame is contiguous
        src = (uint8_t *)in[0].iov_base;

        // Do CSC of complete fram
        // Calculate grid size.
        // Processing two pixels per thread
        dim3 GridSz(DIVUP(in[0].iov_len, BLOCK_W * 2), 1, 1);

        YCrCb422ToRGB_8bit_kernel<<<GridSz, BlockSz>>>(src, dst,in[0].iov_len);
    } else { // num == 2
        // Frame is non-continuous
        src = (uint8_t *)in[0].iov_base;

        // Do CSC and move top of frame to beginning of output buffer
        dst_temp = dst;

        // Calculate grid size.
        // Processing two pixels per thread
        dim3 GridSz1(DIVUP(in[0].iov_len, BLOCK_W * 2), 1, 1);
        YCrCb422ToRGB_8bit_kernel<<<GridSz1, BlockSz>>>(src, dst,in[0].iov_len);

        src = (uint8_t *)in[1].iov_base;

        index = in[0].iov_len; // the source length is converted to Dest length. every 5 bytes converted to 2 pixeles
        // dst = &dst_temp[index]; // Each 5 bytes are converted to two RGB. i.e for HD= [W]1920 * [H]1080 * 3 (each cell is holding 3 bytes for RGB)

        // Do CSC and move bottom of frame to end of output buffer

        // Calculate grid size.
        // Processing two pixels per thread
        dim3 GridSz2(DIVUP(in[1].iov_len, BLOCK_W * 2), 1, 1);
        YCrCb422ToRGB_8bit_kernel<<<GridSz2, BlockSz>>>(src, &dst_temp[index],in[1].iov_len);
    }

    checkCudaErrors(cudaGetLastError());
}

#endif

extern "C"

void YCrCbToRGB_8bit_split(uint8_t* in1, uint8_t* in2, uint32_t size1, uint32_t size2, uchar4 * dst, uint32_t num)
{
#define BLOCK_W 16
#define BLOCK_H 16

    uint8_t* src;
    uchar4* dst_temp;
    uint32_t index = 0;

    // Calculate block size
    dim3 BlockSz(BLOCK_W, BLOCK_H, 1);

    if (num == 1) {

        // Frame is contiguous
        src = (uint8_t*)in1;

        // Do CSC of complete fram
        // Calculate grid size.
        // Processing two pixels per thread
        dim3 GridSz(DIVUP(size1, BLOCK_W * 2), 1, 1);

        YCrCb422ToRGB_8bit_kernel << <GridSz, BlockSz >> > (src, dst, size1);
    }
    else { // num == 2
        // Frame is non-continuous
        src = (uint8_t*)in1;

        // Do CSC and move top of frame to beginning of output buffer
        dst_temp = dst;

        // Calculate grid size.
        // Processing two pixels per thread
        dim3 GridSz1(DIVUP(size1, BLOCK_W * 2), 1, 1);
        YCrCb422ToRGB_8bit_kernel << <GridSz1, BlockSz >> > (src, dst, size1);

        src = (uint8_t*)in2;

        index = size1; // the source length is converted to Dest length. every 5 bytes converted to 2 pixeles
        // dst = &dst_temp[index]; // Each 5 bytes are converted to two RGB. i.e for HD= [W]1920 * [H]1080 * 3 (each cell is holding 3 bytes for RGB)

        // Do CSC and move bottom of frame to end of output buffer

        // Calculate grid size.
        // Processing two pixels per thread
        dim3 GridSz2(DIVUP(size2, BLOCK_W * 2), 1, 1);
        YCrCb422ToRGB_8bit_kernel << <GridSz2, BlockSz >> > (src, &dst_temp[index], size2);
    }

    checkCudaErrors(cudaGetLastError());
}


#endif // #ifndef _CSC_KERNEL_CU_

