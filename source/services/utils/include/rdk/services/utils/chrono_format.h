/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef RDK_SERVICES_UTILS_CHRONO_FORMAT_H_
#define RDK_SERVICES_UTILS_CHRONO_FORMAT_H_

#include <chrono>
#include <ostream>

namespace std
{
namespace chrono
{

template <typename Rep, typename Period>
inline std::ostream& operator<<(std::ostream& os, const duration<Rep, Period>& dur)
{
    os << dur.count() << " [" << Period::num << "/" << Period::den << " s]";
    return os;
}

template <typename Rep>
inline std::ostream& operator<<(std::ostream& os, const duration<Rep, std::nano>& dur)
{
    os << dur.count() << " [ns]";
    return os;
}

template <typename Rep>
inline std::ostream& operator<<(std::ostream& os, const duration<Rep, std::micro>& dur)
{
    os << dur.count() << " [us]";
    return os;
}

template <typename Rep>
inline std::ostream& operator<<(std::ostream& os, const duration<Rep, std::milli>& dur)
{
    os << dur.count() << " [ms]";
    return os;
}

template <typename Rep>
inline std::ostream& operator<<(std::ostream& os, const duration<Rep, std::ratio<1>>& dur)
{
    os << dur.count() << " [s]";
    return os;
}

template <typename Rep>
inline std::ostream& operator<<(std::ostream& os, const duration<Rep, std::ratio<60>>& dur)
{
    os << dur.count() << " [m]";
    return os;
}

template <typename Rep>
inline std::ostream& operator<<(std::ostream& os, const duration<Rep, std::ratio<3600>>& dur)
{
    os << dur.count() << " [h]";
    return os;
}

} // namespace chrono
} // namespace std

#endif // RDK_SERVICES_UTILS_CHRONO_FORMAT_H_
