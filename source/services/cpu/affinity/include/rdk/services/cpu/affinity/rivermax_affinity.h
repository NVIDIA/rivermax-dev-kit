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

#ifndef RDK_SERVICES_CPU_AFFINITY_RIVERMAX_AFFINITY_H_
#define RDK_SERVICES_CPU_AFFINITY_RIVERMAX_AFFINITY_H_

#include "rdk/services/cpu/affinity/rivermax_os_affinity.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
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

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_CPU_AFFINITY_RIVERMAX_AFFINITY_H_ */
