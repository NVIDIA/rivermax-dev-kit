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
#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream> 
#include <vector>
#include <math.h>
#include <thread>

// For shared memory segment
#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>
#endif
#include <fcntl.h>

// CUDA includes
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <inc/helper_cuda.h>
#include <vector_types.h>
