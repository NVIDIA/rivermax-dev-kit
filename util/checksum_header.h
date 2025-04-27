/*
 * Copyright © 2021-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef _CHECKSUM_HEADER_H_
#define _CHECKSUM_HEADER_H_

/**
 * A header that is used by the generic sender and receiver to
 * perform sequence (for dropped packets) and checksum checking.
 */
struct ChecksumHeader
{
    uint32_t sequence;
    uint32_t checksum;
};

#endif // _CHECKSUM_HEADER_H_
