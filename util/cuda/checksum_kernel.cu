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
#include <stdio.h>

static const int blockSize = 1024;

__global__ void 
cuda_compare_checksum_kernel(unsigned int expected, unsigned char* data,
                             unsigned int size, unsigned int* mismatches)
{
    int idx = threadIdx.x;

    // Calculate the sum for each thread.
    int sum = 0;
    for (int i = idx; i < size; i += blockSize)
        sum += data[i];

    __shared__ unsigned int accum[blockSize];
    accum[idx] = sum;

    // Reduce the sums of all blocks.
    __syncthreads();
    for (int size = blockSize / 2; size > 0; size /= 2) {
        if (idx < size)
            accum[idx] += accum[idx + size];
        __syncthreads();
    }

    // Output the results in the first thread.
    if (idx == 0 && accum[0] != expected)
        *mismatches = *mismatches + 1;
}

extern "C"
void cuda_compare_checksum(unsigned int expected, unsigned char* data,
                           unsigned int size, unsigned int* mismatches)
{
    cuda_compare_checksum_kernel<<<1, blockSize>>>(expected, data, size, mismatches);
}
