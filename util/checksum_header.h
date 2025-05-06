/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
