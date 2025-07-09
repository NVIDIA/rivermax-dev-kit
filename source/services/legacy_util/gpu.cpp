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

#include <cerrno>
#include <iostream>
#include <cstring>
#include "defs.h"
#include "gpu.h"

#ifdef CUDA_ENABLED
#ifndef TEGRA_ENABLED
#include <nvml.h>
#endif
#include "rt_threads.h"
#include "checksum_kernel.h"

const gpu_stream DEFAULT_GPU_STREAM = {0};

/**
 * @brief: Initialize GPU.
 *
 * @warning This must be called before any other GPU functions call
 *
 * @param [in] gpu_id     : GPU id.
 *
 * @return: Return status of the operation.
 */
bool gpu_init(int gpu_id)
{
    int ret = -1;
    // nvidia-smi is the user's primary tool for identifying the ID of the GPU which is to be used by the Rivermax application.
    // The ID returned by nvidia-smi is obtained by its enumerating the GPU devices according to their PCI order.
    // However, by default the CUDA driver and runtime APIs enumerate GPUs according to their speed (and not by PCI order).
    // To align the ID obtained from nvidia-smi and those used by the CUDA driver and runtime APIs we modify their GPU ID enumeration
    // policy by setting the CUDA_DEVICE_ORDER environment variable with value PCI_BUS_ID.
    ret = set_enviroment_variable(CUDA_DEVICE_ORDER, CUDA_PCI_BUS_ID_DEVICE_ORDER);
    if (ret != 0) {
        std::cerr << "Failed to set env variable " << CUDA_DEVICE_ORDER << "="
            << CUDA_PCI_BUS_ID_DEVICE_ORDER << std::endl;
        return false;
    }
    std::cout << "Set env variable " << CUDA_DEVICE_ORDER << "="
        << CUDA_PCI_BUS_ID_DEVICE_ORDER << std::endl;

    if (!verify_gpu_device_id(gpu_id)) {
        return false;
    }

    return true;
}

/**
 * @brief: Uninitialize GPU.
 * @warning This must be called in the end of using GPU.
 *
 * @param [in] gpu_id     : GPU id.
 *
 * @return: Return status of the operation.
 */
bool gpu_uninit(int gpu_id)
{
    return true;
}

void gpu_compare_checksum(const uint8_t** data_ptrs, const size_t* sizes,
                          const uint32_t* expected_checksums, uint32_t* mismatch_counter,
                          uint32_t num_packet)
{
    cuda_compare_checksum(data_ptrs, sizes, expected_checksums, mismatch_counter, num_packet);
}

uint32_t* gpu_allocate_counter()
{
    uint32_t *counter;
    cudaMalloc(&counter, sizeof(uint32_t));
    cudaMemset(counter, 0, sizeof(uint32_t));
    return counter;
}

uint32_t gpu_read_counter(uint32_t *counter)
{
    uint32_t result;
    cudaMemcpy(&result, counter, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return result;
}

void gpu_reset_counter(uint32_t *counter)
{
    unsigned int zero = 0;
    cudaMemcpy(counter, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
}

/**
 * @brief: Allocates GPU memory, support both Tegra (Page pinned shared memory)
 *         and non Tegara (Device memory) GPUs.
 *
 * @param [in] gpu_id: GPU id.
 * @param [in] size  : Size of the memory to allocate.
 * @param [in] align : Alignment of the reserved virtual address size requested.
 *
 * @return: Pointer to the allocated memory.
 */
void* gpu_allocate_memory(int gpu_id, size_t size, size_t align)
{
    int count;
    cudaError_t cuda_err = cudaGetDeviceCount(&count);
    if (cuda_err != cudaSuccess || count <= gpu_id) {
        std::cerr << "Failed to allocate GPU memory; GPU " << gpu_id << " not available." << std::endl;
        return nullptr;
    }

    cudaDeviceProp props;
    cuda_err = cudaGetDeviceProperties(&props, gpu_id);
    if (cuda_err != cudaSuccess || !props.canMapHostMemory) {
        std::cerr << "Failed to allocate GPU memory; host mapping not supported." << std::endl;
        return nullptr;
    }

    cuda_err = cudaSetDevice(gpu_id);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory; failed to set device." << std::endl;
        return nullptr;
    }

#ifdef TEGRA_ENABLED
    cuda_err = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory; failed to set device flags." << std::endl;
        return nullptr;
    }
#endif

    char* buffer;
#ifdef TEGRA_ENABLED
    cuda_err = cudaMallocHost((void**)&buffer, size);
#else
    buffer = (char*)cudaAllocateMmap(gpu_id, size, align);
#endif
    if (cuda_err != cudaSuccess || buffer == nullptr) {
        std::cerr << "Failed to allocate GPU memory." << std::endl;
        return nullptr;
    }

#ifndef TEGRA_ENABLED
    unsigned int flag = 1;
    cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)buffer);
#endif
    cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory; failed to synchronize. error: " << cuda_err << std::endl;
        return nullptr;
    }

    std::cout << "GPU allocation succeeded, GPU id = " << gpu_id << " ,size = " << size << std::endl;
    return buffer;
}

/**
 * @brief: Free GPU memory, support both Tegra (Page pinned shared memory)
 *         and non Tegara (Device memory) GPUs.
 *
 * @param [in] ptr: Memory address to free
 *
 * @return: Return status of the operation.
 */
bool gpu_free_memory(void* ptr, size_t size)
{
#ifdef TEGRA_ENABLED
    NOT_IN_USE(size);
    cudaError_t cuda_err = cudaFreeHost (ptr);
    if (cuda_err != cudaSuccess && cuda_err != cudaErrorCudartUnloading) {
        std::cerr << "Failed to free GPU memory, ret " << cuda_err << std::endl;
        return false;
    }
#else
     CUresult cuda_err = cudaFreeMmap((uint64_t*)&ptr, size);
     if (cuda_err != CUDA_SUCCESS && cuda_err != CUDA_ERROR_DEINITIALIZED) {
        std::cerr << "Failed to free GPU memory, ret " << cuda_err << std::endl;
        return false;
     }
#endif
    return true;    
}

size_t gpu_query_alignment(int gpu_id)
{
#ifdef TEGRA_ENABLED
    return 1;
#else // TEGRA_ENABLED
    CUresult status = CUDA_SUCCESS;
    size_t granularity = 0;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = gpu_id;
    // Get the minimum granularity needed for the resident devices
    // (the max of the minimum granularity of each participating device)
    status = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (status != CUDA_SUCCESS) {
        std::cout << "cuMemGetAllocationGranularity failed status = " << status << "\n";
        return 1;
    }
    return granularity;
#endif
}

size_t gpu_align_physical_allocation_size(int gpu_id, size_t allocation_size)
{
    size_t size = allocation_size;
    size_t granularity = gpu_query_alignment(gpu_id);
    // Round up the size such that we can evenly split it into a stripe size that
    // meets the granularity requirements Essentially size = N *
    // residentDevices.size() * min_granularity is the requirement, since each
    // piece of the allocation will be stripeSize = N * min_granularity and the
    // min_granularity requirement applies to each stripeSize piece of the
    // allocation.
    size = round_up(allocation_size, granularity); /* This must always co-exist with the NIC size restrictions. Is it guaranteed to? */
    return size;
}

#ifndef TEGRA_ENABLED
void* cudaAllocateMmap(int gpu_id, size_t size, size_t align)
{
    CUresult status = CUDA_SUCCESS;
    CUdeviceptr dptr = 0;
    int val = 0;
    std::cout << "CUDA memory allocation on GPU - cuMemCreate " << std::endl;

    // Setup the properties common for all the chunks
    // The allocations will be device pinned memory.
    // This property structure describes the physical location where the memory
    // will be allocated via cuMemCreate along with additional properties In this
    // case, the allocation will be pinned device memory local to a given device.
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = gpu_id;

    status = cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, prop.location.id);
    if (status != CUDA_SUCCESS || val == 0) {
        std::cout << "Device does not support VA. status = " << status << "\n";
        goto done;
    }

    status = cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, prop.location.id);
    if (status != CUDA_SUCCESS || val == 0) {
        std::cout << "RDMA is not supported or not enabled, status = " << status << " val = " << val << "\n";
        goto done;
    } else {
        std::cout << "RDMA is supported and enabled, status \n";
        prop.allocFlags.gpuDirectRDMACapable = 1;
    }

    // Reserve the required contiguous VA space for the allocations
    status = cuMemAddressReserve(&dptr, size, align, 0, 0);
    if (status != CUDA_SUCCESS) {
        std::cout << "cuMemAddressReserve failed status = " << status << "\n";
        goto done;
    }

    // Create the allocation as a pinned allocation on this device
    CUmemGenericAllocationHandle allocationHandle;
    status = cuMemCreate(&allocationHandle, size, &prop, 0);
    if (status != CUDA_SUCCESS) {
        std::cout << "cuMemCreate failed status = " << status << "\n";
        goto done;
    }

    // Assign the chunk to the appropriate VA range and release the handle.
    // After mapping the memory, it can be referenced by virtual address.
    // Since we do not need to make any other mappings of this memory or export
    // it, we no longer need and can release the allocationHandle. The
    // allocation will be kept live until it is unmapped.
    status = cuMemMap(dptr, size, 0, allocationHandle, 0);

    // the handle needs to be released even if the mapping failed.
    status = cuMemRelease(allocationHandle);
    if (status != CUDA_SUCCESS) {
        std::cout << "cuMemRelease failed status = " << status << "\n";
        goto done;
    }
    // Each accessDescriptor will describe the mapping requirement for a single
    // device
    CUmemAccessDesc accessDescriptors;

    // Prepare the access descriptor array indicating where and how the backings
    // should be visible.
    // Specify which device we are adding mappings for.
    accessDescriptors.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    accessDescriptors.location.id = gpu_id;

    // Specify both read and write access.
    accessDescriptors.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Apply the access descriptors to the whole VA range.
    status = cuMemSetAccess(dptr, size, &accessDescriptors, 1);
    if (status != CUDA_SUCCESS) {
        std::cout << "cuMemSetAccess failed status = " << status << "\n";
        goto done;
    }

    std::cout << "CUDA memory allocation on GPU - cuMemCreate Done" << std::endl;

done:
    if (status != CUDA_SUCCESS) {
        CUresult free_status = cudaFreeMmap((uint64_t*)&dptr, size);
        std::cout << "CUDA memory free finished with status " << free_status << std::endl;
        return nullptr;
    }

    return (void*)dptr;
}

CUresult cudaFreeMmap(uint64_t *ptr, size_t size)
{
    if (!ptr) {
        return CUDA_SUCCESS;
    }
    std::cout << "CUDA cudaFreeMmap " << std::hex << *ptr << std::dec << std::endl;
    CUdeviceptr dptr = *(CUdeviceptr*)ptr;
    CUresult status = CUDA_SUCCESS;

    // Unmap the mapped virtual memory region
    // Since the handles to the mapped backing stores have already been released
    // by cuMemRelease, and these are the only/last mappings referencing them,
    // The backing stores will be freed.
    // Since the memory has been unmapped after this call, accessing the specified
    // va range will result in a fault (until it is re-mapped).
    status = cuMemUnmap(dptr, size);
    if (status != CUDA_SUCCESS) {
        return status;
    }
    // Free the virtual address region.  This allows the virtual address region
    // to be reused by future cuMemAddressReserve calls.  This also allows the
    // virtual address region to be used by other allocation made through
    // Operating system calls like malloc & mmap.
    status = cuMemAddressFree(dptr, size);
    return status;
}
#endif

/**
 * @brief: Set GPU memory, support both Tegra (Page pinned shared memory)
 *         and non Tegara (Device memory) GPUs.
 *
 * @param [in] dst: Destination memory address.
 * @param [in] value: Value to set for each byte of specified memory.
 * @param [in] count: Size in bytes to set.
 *
 * @return: Return status of the operation.
 */
bool gpu_memset(void* dst, int value, size_t count)
{
    cudaError_t cuda_err = cudaSuccess;

#ifdef TEGRA_ENABLED
    std::memset(dst, value, count);
#else
    cuda_err = cudaMemset(dst, value, count);
#endif
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to set GPU memory." << std::endl;
        return false;
    }

    return true;
}

/**
 * @brief: Convert gpu_memcpy_direction to cudaMemcpyKind.
 *
 * @param [in] direction: GPU memory copy direction
 *
 * @return: Equivalent cudaMemcpyKind value.
 */
cudaMemcpyKind to_cuda_memcpy_kind(gpu_memcpy_direction direction)
{
    switch (direction) {
        case gpu_memcpy_direction::gpuMemcpyHostToHost:
            return cudaMemcpyHostToHost;
        case gpu_memcpy_direction::gpuMemcpyHostToDevice:
            return cudaMemcpyHostToDevice;
        case gpu_memcpy_direction::gpuMemcpyDeviceToHost:
            return cudaMemcpyDeviceToHost;
        case gpu_memcpy_direction::gpuMemcpyDeviceToDevice:
            return cudaMemcpyDeviceToDevice;
        case gpu_memcpy_direction::gpuMemcpyDefault:
        default:
            return cudaMemcpyDefault;
    }
}
/**
 * @brief: Copy to/from GPU memory.
 *
 * Copy to/from GPU memory, support both Tegra (Page pinned shared memory)
 * and non Tegra (Device memory) GPUs.
 *
 * @param [in] dst: Destination memory address.
 * @param [in] src: Source memory address.
 * @param [in] count: Size in bytes to copy.
 * @param [in] direction: Direction of copy operation.
 * @param [in] stream: GPU stream to use for the copy operation.
 * @param [in] sync_mode: Whether to synchronize after the copy.
 *
 * @return: Return status of the operation.
 */
bool gpu_memcpy(void* dst, const void* src, size_t count,
    gpu_memcpy_direction direction, gpu_stream stream, gpu_sync_mode sync_mode)
{
    cudaError_t cuda_err = cudaSuccess;

#ifdef TEGRA_ENABLED
    std::memcpy(dst, src, count);
#else
    cuda_err = cudaMemcpyAsync(dst, src, count, to_cuda_memcpy_kind(direction), stream.cuda_stream);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to copy memory GPU memory." << std::endl;
        return false;
    }

    if (sync_mode == gpu_sync_mode::SYNC) {
        if (!gpu_synchronize_stream(stream)) {
            return false;
        }
    }
#endif
    return true;
}
/**
 * @brief: Copies a matrix.
 *
 * Copies a matrix (@p height rows of @p width bytes each) from the memory area
 * pointed to by @p src to the memory area pointed to by @p dst.
 * @p dst_padded_width and @p src_padded_width are the widths in memory in bytes
 * of the 2D arrays pointed to by @p dst and @p src, including any padding added
 * to the end of each row.
 *
 * @note: The memory areas may not overlap.
 * @note: @p width must not exceed either @p dst_padded_width or @p src_padded_width.
 *
 * @param [in] dst: Destination memory address.
 * @param [in] dst_padded_width: Padded memory width of destination memory.
 * @param [in] src: Source memory address.
 * @param [in] src_padded_width: Padded memory width of source memory.
 * @param [in] width: Width of matrix transfer (columns in bytes).
 * @param [in] height: Height of matrix transfer (rows).
 * @param [in] direction: Direction of copy operation.
 * @param [in] stream: CUDA stream to use for the copy operation.
 * @param [in] sync_mode: Whether to synchronize after the copy.
 *
 * @return: Status of the operation.
 */
bool gpu_memcopy_2D(void* dst, size_t dst_padded_width,
    const void* src, size_t src_padded_width, size_t width, size_t height,
    gpu_memcpy_direction direction, gpu_stream stream, gpu_sync_mode sync_mode)
{
#ifdef TEGRA_ENABLED
    host_mem_copy_2D(dst, dst_padded_width, src, src_padded_width, width, height);
#else
    if (!dst || !src || width == 0 || height == 0 || dst_padded_width == 0 ||
        src_padded_width == 0 || (width > dst_padded_width) || (width > src_padded_width)) {
        std::cerr << "Invalid parameters for 2D memory copy" << std::endl;
        return false;
    }
    cudaError_t cuda_result = cudaMemcpy2DAsync(dst, dst_padded_width,
        src, src_padded_width, width, height, to_cuda_memcpy_kind(direction), stream.cuda_stream);
    if (cuda_result != cudaSuccess) {
        std::cerr << "Failed to copy 2D memory on GPU : "
            << cudaGetErrorString(cuda_result) << std::endl;
        std::cerr << "dst: " << dst << ", dst_padded_width: " << dst_padded_width
            << ", src: " << src << ", src_padded_width: " << src_padded_width
            << ", width: " << width << ", height: " << height << std::endl;
        return false;
    }

    if (sync_mode == gpu_sync_mode::SYNC) {
        if (!gpu_synchronize_stream(stream)) {
            return false;
        }
    }
#endif
    return true;
}
/**
 * @brief: Synchronize GPU stream.
 *
 * Wait for all operations in the specified stream to complete.
 *
 * @param [in] stream: GPU stream to synchronize.
 *
 * @return: Status of the operation.
 */
bool gpu_synchronize_stream(gpu_stream stream)
{
    cudaError_t cuda_err = cudaStreamSynchronize(stream.cuda_stream);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to synchronize GPU stream: " << cudaGetErrorString(cuda_err) << std::endl;
        return false;
    }
    return true;
}

bool gpu_create_stream(gpu_stream* stream)
{
    if (stream == nullptr) {
        std::cerr << "Failed to create GPU stream; stream is null." << std::endl;
        return false;
    }
    cudaError_t cuda_result = cudaStreamCreate(&stream->cuda_stream);
    if (cuda_result != cudaSuccess) {
        std::cerr << "Failed to create GPU stream: "
            << cudaGetErrorString(cuda_result) << std::endl;
        return false;
    }
    return true;
}

bool gpu_destroy_stream(gpu_stream stream)
{
    cudaError_t cuda_result = cudaStreamDestroy(stream.cuda_stream);
    if (cuda_result != cudaSuccess && cuda_result != cudaErrorCudartUnloading) {
        std::cerr << "Failed to destroy CUDA stream: "
            << cudaGetErrorString(cuda_result) << std::endl;
        return false;
    }
    return true;
}

bool verify_gpu_device_id(int device_id)
{
    int count = 0;

    cudaGetDeviceCount(&count);
    // verify user configuration is correct
    if (device_id != GPU_ID_INVALID) {
        if ((device_id >= count) || (0 > device_id)) {
            std::cout << "ERROR: User set the GPU id as = " << device_id << " but maximum allowed GPU id is " << count - 1 << std::endl;
            return false;
        }
        const std::string gpu_name = get_gpu_device_name(device_id);
        std::cout << "Using GPU: " << gpu_name << " with id = " << device_id << std::endl;
    }
    return true;
}

const std::string get_gpu_device_name(int device_id)
{
    cudaDeviceProp prop;
    std::string device_name = "";

    if (device_id != GPU_ID_INVALID) {
        cudaGetDeviceProperties(&prop, device_id);
        device_name = prop.name;
    }

    return device_name;
}

bool set_gpu_device(int gpu_id)
{
    cudaError_t cuda_err = cudaSuccess;

    cuda_err = cudaSetDevice(gpu_id);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to set gpu device." << std::endl;
        return false;
    }

    return true;
}

#ifndef TEGRA_ENABLED
/**
 * @brief: Sets GPU and memory clocks to locked on max frequency
 *
 * @param [in] gpu_id: GPU id
 *
 * @return: Return status of the operation:
 *           0       - in case of success,
 *          -ENOTSUP - when locked clocks are not supported,
 *          -EPERM   - in case of other errors.
 */
int gpu_set_locked_clocks_max_freq(int gpu_id)
{
    nvmlReturn_t nvret = NVML_SUCCESS;
    int ret = 0;
    uint32_t max_graphics_clock_freq = 0;
    uint32_t max_memory_clock_freq = 0;

    nvret = nvmlInit();
    if (nvret != NVML_SUCCESS) {
        std::cerr << "Failed to init The NVIDIA Management Library (NVML) with error: "
            << nvret << std::endl;
        return -EPERM;
    }

    nvmlDevice_t nvDevice;
    nvret = nvmlDeviceGetHandleByIndex(gpu_id, &nvDevice);
    if (nvret != NVML_SUCCESS) {
        std::cerr << "Failed to get nvmlDevice with error: " << nvret << std::endl;
        ret = -EPERM;
        goto recover;
    }

    nvret = nvmlDeviceGetMaxClockInfo(nvDevice, NVML_CLOCK_GRAPHICS, &max_graphics_clock_freq);
    if (nvret != NVML_SUCCESS) {
        std::cerr << "Failed to get max graphics clock frequency with error: " << nvret << std::endl;
        ret = -EPERM;
        goto recover;
    }

    nvret = nvmlDeviceGetMaxClockInfo(nvDevice, NVML_CLOCK_MEM, &max_memory_clock_freq);
    if (nvret != NVML_SUCCESS) {
        std::cerr << "Failed to get max memory clock frequency with error: " << nvret << std::endl;
        ret = -EPERM;
        goto recover;
    }

    nvret = nvmlDeviceSetGpuLockedClocks(nvDevice, max_graphics_clock_freq, max_graphics_clock_freq);
    if (nvret != NVML_SUCCESS) {
        if (nvret == NVML_ERROR_NOT_SUPPORTED) {
            std::cout << "Warning! Setting locked gpu clock is not supported" << std::endl;
            ret = -ENOTSUP;
        } else {
            std::cerr << "Failed to set gpu clock on max frequency with error: " << nvret << std::endl;
            ret = -EPERM;
        }
        goto recover;
    }

    nvret = nvmlDeviceSetMemoryLockedClocks(nvDevice, max_memory_clock_freq, max_memory_clock_freq);
    if (nvret != NVML_SUCCESS) {
        if (nvret == NVML_ERROR_NOT_SUPPORTED) {
            std::cout << "Warning! Setting locked memory clock is not supported" << std::endl;
            ret = -ENOTSUP;
        } else {
            std::cerr << "Failed to set memory clock on max frequency with error: " << nvret << std::endl;
            ret = -EPERM;
        }
        goto recover;
    }

    goto cleanup;

recover:
    nvmlDeviceResetGpuLockedClocks(nvDevice);
    nvmlDeviceResetMemoryLockedClocks(nvDevice);

cleanup:
    nvret = nvmlShutdown();
    if (nvret != NVML_SUCCESS) {
        std::cerr << "Failed to shutdown The NVIDIA Management Library (NVML) with error: "
            << nvret << std::endl;
        return -EPERM;
    }
    return ret;
}

/**
 * @brief: Reset GPU and memory clocks to locked on default frequency
 *
 * @param [in] gpu_id: GPU id
 *
 * @return: Return status of the operation:
 *           0       - in case of success,
 *          -ENOTSUP - when locked clocks are not supported,
 *          -EPERM   - in case of other errors.
 */
int gpu_reset_locked_clocks(int gpu_id)
{
    nvmlReturn_t nvret = NVML_SUCCESS;
    int ret = 0;

    nvret = nvmlInit();
    if (nvret != NVML_SUCCESS) {
        std::cerr << "Failed to init The NVIDIA Management Library (NVML) with error: "
            << nvret << std::endl;
        return -EPERM;
    }

    nvmlDevice_t nvDevice;
    nvret = nvmlDeviceGetHandleByIndex(gpu_id, &nvDevice);
    if (nvret != NVML_SUCCESS) {
        std::cerr << "Failed to get nvmlDevice with error: " << nvret << std::endl;
        ret = -EPERM;
        goto cleanup;
    }

    nvret = nvmlDeviceResetGpuLockedClocks(nvDevice);
    if (nvret != NVML_SUCCESS) {
        if (nvret == NVML_ERROR_NOT_SUPPORTED) {
            std::cout << "Warning! Resetting locked gpu clock is not supported" << std::endl;
            ret = -ENOTSUP;
        } else {
            std::cerr << "Failed to reset gpu clock on default frequency with error: " << nvret << std::endl;
            ret = -EPERM;
        }
        goto cleanup;
    }

    nvret = nvmlDeviceResetMemoryLockedClocks(nvDevice);
    if (nvret != NVML_SUCCESS) {
        if (nvret == NVML_ERROR_NOT_SUPPORTED) {
            std::cout << "Warning! Resetting locked gpu clock is not supported" << std::endl;
            ret = -ENOTSUP;
        } else {
            std::cerr << "Failed to reset memory clock on default frequency with error: " << nvret << std::endl;
            ret = -EPERM;
        }
    }

cleanup:
    nvret = nvmlShutdown();
    if (nvret != NVML_SUCCESS) {
        std::cerr << "Failed to shutdown The NVIDIA Management Library (NVML) with error: "
            << nvret << std::endl;
        return false;
    }
    return ret;
}

/**
 * @brief: Query GPU BAR1 memory information
 *
 * @param [in] gpu_id   : GPU id.
 * @param [out] mem_info: GPU BAR1 memory information @ref gpu_bar1_memory_info
 *
 * @return: Return status of the operation.
 */
bool gpu_query_bar1_memory_info(int gpu_id, gpu_bar1_memory_info& mem_info)
{
    nvmlReturn_t nvret = NVML_SUCCESS;
    bool ret = false;

    nvret = nvmlInit();
    if (nvret != NVML_SUCCESS) {
        std::cerr << "Failed to init The NVIDIA Management Library (NVML) with error: "
            << nvret << std::endl;
        return false;
    }

    nvmlDevice_t nvDevice;
    nvret = nvmlDeviceGetHandleByIndex(gpu_id, &nvDevice);
    if (nvret != NVML_SUCCESS) {
        std::cerr << "Failed to get nvmlDevice with error: " << nvret << std::endl;
        goto end;
    }

    nvmlBAR1Memory_t nvBarMemory;
    nvret = nvmlDeviceGetBAR1MemoryInfo(nvDevice, &nvBarMemory);
    if (nvret != NVML_SUCCESS) {
        std::cerr << "Failed to get GPU BAR1 memory information with error: " << nvret << std::endl;
        goto end;
    }
    mem_info.free = nvBarMemory.bar1Free;
    mem_info.total = nvBarMemory.bar1Total;
    mem_info.used = nvBarMemory.bar1Used;
    ret = true;

end:
    nvret = nvmlShutdown();
    if (nvret != NVML_SUCCESS) {
        std::cerr << "Failed to shutdown The NVIDIA Management Library (NVML) with error: "
            << nvret << std::endl;
        return false;
    }
    return ret;
}

/**
 * @brief: Verify that BAR1 has enough memory to allocate.
 *
 * @param [in] gpu_id: GPU id
 * @param [in] size  : size to compare with free size on BAR1
 *
 * @return: Return status of the operation.
 */
bool gpu_verify_allocated_bar1_size(int gpu_id, size_t size)
{
    gpu_bar1_memory_info bar1_mem_info;
    memset(&bar1_mem_info, 0, sizeof(bar1_mem_info));

    if (!gpu_query_bar1_memory_info(gpu_id, bar1_mem_info)) {
        std::cerr << "Failed to query GPU BAR1 memory information." << std::endl;
        return false;
    }

    if (size > bar1_mem_info.free) {
        std::cerr << "There is no enough BAR1 memory on the GPU; maximum size is " << bar1_mem_info.total
            << " MB. Unallocated free memory is " << bar1_mem_info.free << " MB (requested "
            << size << ")" << std::endl;
        return false;
    }
    return true;
}
#endif

void* gpu_allocate_host_pinned_memory(size_t size)
{
    void* mem_ptr = nullptr;
    cudaError_t error = cudaMallocHost(&mem_ptr, size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate " << size <<
        " bytes of GPU host pinned memory with error: " << cudaGetErrorString(error) << std::endl;
        return nullptr;
    }
    return mem_ptr;
}

bool gpu_free_host_pinned_memory(void* ptr)
{
    if (ptr == nullptr) {
        std::cerr << "Failed to free the pointer at address " << ptr << std::endl;
        return false;
    }

    cudaError_t error = cudaFreeHost(ptr);
    if (error != cudaSuccess && error != cudaErrorCudartUnloading) {
        std::cerr << "Failed to free GPU host pinned memory with error: "
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }

    return true;
}

#endif // CUDA_ENABLED

/**
 * @brief: Copies a matrix. (Host version)
 *
 * Copies a matrix (@p height rows of @p width bytes each) from the memory area
 * pointed to by @p src to the memory area pointed to by @p dst.
 * @p dst_padded_width and @p src_padded_width are the widths in memory in bytes
 * of the 2D arrays pointed to by @p dst and @p src, including any padding added
 * to the end of each row.
 *
 * @note: The memory areas may not overlap.
 * @note: @p width must not exceed either @p dst_padded_width or @p src_padded_width.
 *
 * @param [in] dst: Destination memory address.
 * @param [in] dst_padded_width: Padded memory width of destination memory.
 * @param [in] src: Source memory address.
 * @param [in] src_padded_width: Padded memory width of source memory.
 * @param [in] width: Width of matrix transfer (columns in bytes).
 * @param [in] height: Height of matrix transfer (rows).
 *
 * @return: Status of the operation.
 */
bool host_mem_copy_2D(void* dst, size_t dst_padded_width,
    const void* src, size_t src_padded_width, size_t width, size_t height)
{
    if (!dst || !src || width == 0 || height == 0 || dst_padded_width == 0 ||
        src_padded_width == 0 || (width > dst_padded_width) || (width > src_padded_width)) {
        std::cerr << "Invalid parameters for host_memory_copy_2D" << std::endl;
        return false;
    }

    uint8_t* dst_ptr = static_cast<uint8_t*>(dst);
    const uint8_t* src_ptr = static_cast<const uint8_t*>(src);

    for (size_t row = 0; row < height; ++row) {
        uint8_t* dst_row = dst_ptr + row * dst_padded_width;
        const uint8_t* src_row = src_ptr + row * src_padded_width;
        std::memcpy(dst_row, src_row, width);
    }
    return true;
}
