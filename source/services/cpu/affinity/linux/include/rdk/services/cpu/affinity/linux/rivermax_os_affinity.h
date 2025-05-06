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

#ifndef RDK_SERVICES_CPU_AFFINITY_LINUX_RIVERMAX_OS_AFFINITY_H_
#define RDK_SERVICES_CPU_AFFINITY_LINUX_RIVERMAX_OS_AFFINITY_H_

#include <thread>

#include <sys/sysinfo.h>

#include <rivermax_api.h> // IWYU pragma: export

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Linux-specific implementation of the CPU affinity management.
 */
class LinuxAffinity
{
public:
    /**
     * @brief: Interface for the Linux-specific OS API.
     */
    class os_api
    {
    public:
        virtual ~os_api() = default;
        /**
         * @brief: Returns the number of processors available to the system.
         *
         * @return: The number of processors available to the system.
         */
        virtual int get_proc_count() const { return get_nprocs(); }
        /**
         * @brief: Allocates a CPU set.
         *
         * @param [in] count: The number of CPUs to allocate.
         *
         * @return: A pointer to the allocated CPU set.
         */
        virtual cpu_set_t* cpu_alloc(size_t count) const { return CPU_ALLOC(count); }
        /**
         * @brief: Frees the allocated CPU set.
         *
         * @param [in] cpu_set: A pointer to the CPU set to free.
         */
        virtual void cpu_free(cpu_set_t* cpu_set) const { CPU_FREE(cpu_set); }
        /**
         * @brief: Returns the size of the CPU set.
         *
         * @param [in] count: The number of CPUs to allocate.
         *
         * @return: The size of the CPU set.
         */
        virtual size_t cpu_alloc_size(size_t count) const { return CPU_ALLOC_SIZE(count); }
        /**
         * @brief: Clears the CPU set.
         *
         * @param [in] size: The size of the CPU set.
         * @param [in] cpu_set: A pointer to the CPU set to clear.
         */
        virtual void cpu_zero_s(size_t size, cpu_set_t* cpu_set) const { CPU_ZERO_S(size, cpu_set); }
        /**
         * @brief: Sets the CPU set.
         *
         * @param [in] processor: The processor to set.
         * @param [in] cpu_set: A pointer to the CPU set to set.
         */
        virtual void cpu_set(size_t processor, cpu_set_t* cpu_set) const { CPU_SET(processor, cpu_set); }
        /**
         * @brief: Returns the CPU set.
         *
         * @param [in] processor: The processor to check.
         * @param [in] cpu_set: A pointer to the CPU set to check.
         *
         * @return: true if the processor is set, false otherwise.
         */
        virtual std::thread::native_handle_type this_thread_handle() const { return pthread_self(); }
        /**
         * @brief: Returns the CPU set.
         *
         * @param [in] processor: The processor to check.
         * @param [in] cpu_set: A pointer to the CPU set to check.
         *
         * @return: true if the processor is set, false otherwise.
         */
        virtual int set_affinity_np (pthread_t handle, size_t set_size, const cpu_set_t *cpu_set) const
        {
            auto status = pthread_setaffinity_np(handle, set_size, cpu_set);
            return (status == 0)? 0: ((errno != 0) ? errno: -1);
        }
    };

    /**
     * @brief: Editor for the CPU affinity.
     */
    class editor
    {
    public:
        /**
         * @brief: Constructor.
         *
         * @param [in] affinity: The affinity to edit.
         * @param [in] thread: The thread to edit.
         */
        editor(const LinuxAffinity &affinity, std::thread::native_handle_type thread);
        ~editor();
        /**
         * @brief: Sets the processor.
         *
         * @param [in] processor: The processor to set.
         */
        void set(size_t processor);
        /**
         * @brief: Applies the changes.
         */
        void apply();
    protected:
        const os_api &m_os_api;
        std::thread::native_handle_type m_thread;
        cpu_set_t *m_cpu_set;
        size_t m_set_size;
    };
    /**
     * @brief: Constructor.
     *
     * @param [in] os_api: The Linux-specific OS API.
     */
    explicit LinuxAffinity(const os_api& os_api);
    /**
     * @brief: Returns the number of cores.
     */
    size_t count_cores() const;

protected:
    const os_api &m_os_api;
};
/**
 * @brief: Linux-specific implementation of the CPU affinity management.
 */
class OsSpecificAffinity : public LinuxAffinity
{
public:
    OsSpecificAffinity(const os_api &os_api): LinuxAffinity {os_api} {}
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_CPU_AFFINITY_LINUX_RIVERMAX_OS_AFFINITY_H_ */
