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
#pragma once
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <assert.h>

#define ck(call) check(call, __LINE__, __FILE__)

#define SAMPLE_THROW_ERROR(errorString, status) do {                   \
    std::stringstream _error;                                          \
    _error << errorString << " " << status                             \
    << " In function, " << __FUNCTION__ << " In file, " << __FILE__    \
    << " at line, " << __LINE__ << std::endl;                          \
    throw std::runtime_error(_error.str());                            \
} while(0)

#define CK_CUDA(func) do {                                             \
    cudaError_t status = (func);                                       \
    if (status != 0) {                                                 \
        SAMPLE_THROW_ERROR("Cuda Runtime Failure: ", status);           \
    }                                                                  \
} while(0)

#define CK_CUDA_DRV_API(func) do {                                     \
    CUresult status = (func);                                          \
    if (status != 0) {                                                 \
        SAMPLE_THROW_ERROR("Cuda Driver API Failure", status);          \
    }                                                                  \
} while(0)

inline void* cudaAlloc(size_t size) { void* p; CK_CUDA(cudaMalloc((void**)&p, size)); return p; }
struct cudaRelease
{
	void operator()(void* p)
	{
		if (p) cudaFree(p);
		p = nullptr;
	}
};

typedef std::unique_ptr<void, cudaRelease> cudaAllocation;

inline void* memAlloc(size_t size) { void* p; p = malloc(size); return p; }
struct memRelease
{
	void operator()(void* p)
	{
		if (p) free(p);
		p = nullptr;
	}
};

typedef std::unique_ptr<void, memRelease> memAllocation;

