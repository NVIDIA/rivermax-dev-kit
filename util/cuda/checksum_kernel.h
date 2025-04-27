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
// Calculates the checksum of a data packet and compares it to an expected checksum.
extern "C" void cuda_compare_checksum(unsigned int expected, unsigned char* data,
                                      unsigned int size, unsigned int* mismatches);
