/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once
#include "rivermax_os_affinity.h"

namespace rivermax
{
namespace libs
{

class Affinity : public OsSpecificAffinity
{
public:
    using mask = rmax_cpu_set_t;

protected:
    static const os_api default_api;

public:
    Affinity(const os_api &os_api = default_api);
    ~Affinity();
    void set(std::thread &thread, const size_t processor);
    void set(std::thread &thread, const mask &cpu_mask);
    void set(const size_t processor);
    void set(const mask &cpu_mask);
private:
    void fill_with(const mask &cpu_mask, editor &editor);
};

bool set_affinity(const size_t processor) noexcept;

bool set_affinity(const Affinity::mask &cpu_mask) noexcept;

} // namespace libs
} // namespace rivermax
