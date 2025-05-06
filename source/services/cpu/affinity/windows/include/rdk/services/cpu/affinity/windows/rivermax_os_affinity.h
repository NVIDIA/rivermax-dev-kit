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

#ifndef RDK_SERVICES_CPU_AFFINITY_WINDOWS_RIVERMAX_OS_AFFINITY_H_
#define RDK_SERVICES_CPU_AFFINITY_WINDOWS_RIVERMAX_OS_AFFINITY_H_

#include <cstddef>
#include <memory>
#include <thread>

#include <rivermax_api.h> // IWYU pragma: export

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Windows-specific implementation of the CPU affinity management.
 */
class WindowsAffinity
{
public:
    /**
     * @brief: Interface for the Windows-specific OS API.
     */
    class os_api
    {
    public:
        /**
         * @brief: Returns the number of processors available to the system.
         *
         * @return: The number of processors available to the system.
         */
        virtual DWORD get_logical_processor_information_ex(
            const LOGICAL_PROCESSOR_RELATIONSHIP RelationshipType,
            SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* Buffer,
            DWORD* ReturnedLength) const
        {
            return to_error_code(GetLogicalProcessorInformationEx(RelationshipType, Buffer, ReturnedLength));
        }
        /**
         * @brief: Returns the number of processors available to the system.
         *
         * @return: The number of processors available to the system.
         */
        virtual DWORD get_numa_processor_node_ex(PROCESSOR_NUMBER* Processor, USHORT* NodeNumber) const
        {
            return to_error_code(GetNumaProcessorNodeEx(Processor, NodeNumber));
        }
        /**
         * @brief: Sets the affinity of the specified thread.
         *
         * @param [in] hThread: The handle to the thread.
         * @param [in] dwThreadAffinityMask: The affinity mask to set.
         * @param [out] pdwPreviousAffinityMask: The previous affinity mask.
         *
         * @return: The error code.
         */
        virtual DWORD set_thread_group_affinity(
            HANDLE hThread, const GROUP_AFFINITY* GroupAffinity, GROUP_AFFINITY* PreviousGroupAffinity) const
        {
            return to_error_code(SetThreadGroupAffinity(hThread, GroupAffinity, PreviousGroupAffinity));
        }
        /**
         * @brief: Returns the handle of the current thread.
         *
         * @return: The handle of the current thread.
         */
        virtual std::thread::native_handle_type this_thread_handle() const { return ::GetCurrentThread(); }

    protected:
        /**
         * @brief: Converts the return value to the error code.
         *
         * @param [in] return_value: The return value.
         *
         * @return: The error code.
         */
        DWORD to_error_code(BOOL return_value) const
        {
            if (return_value) {
                return 0;
            }

            auto error_code = GetLastError();
            return error_code ? error_code : (DWORD)(-1);
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
        editor(const WindowsAffinity& affinity, std::thread::native_handle_type thread);
        void set(const size_t processor);
        void apply();

    private:
        const WindowsAffinity& m_affinity;
        std::thread::native_handle_type m_thread;
        const PROCESSOR_GROUP_INFO* m_group;
        KAFFINITY m_mask;
        size_t m_1st_processor_in_group;

        /**
         * @brief: Finds the group for the specified processor.
         *
         * @param [in] processor: The processor to find the group for.
         */
        void find_group(size_t processor);
        /**
         * @brief: Determines the group for the specified processor.
         *
         * @param [in] processor: The processor to determine the group for.
         */
        void determine_group(size_t processor);
        /**
         * @brief: Sets the affinity of the specified thread.
         *
         * @param [in] processor: The processor to set the affinity for.
         */
        void set_ingroup_affinity(size_t processor);
        /**
         * @brief: Determines if the specified processor is in the current group.
         *
         * @param [in] processor: The processor to check.
         *
         * @return: true if the processor is in the current group, false otherwise.
         */
        bool is_in_current_group(size_t processor)
        {
            return ((m_1st_processor_in_group + m_group->ActiveProcessorCount) > processor);
        }
    };
    /**
     * @brief: Database for the CPU affinity.
     */
    class database : public SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX
    {
    public:
        /**
         * @brief: Builds the database.
         *
         * @param [in] win_api: The Windows-specific OS API.
         *
         * @return: The database.
         */
        static database* build(const os_api& win_api);
    };

    /**
     * @brief: Constructor.
     *
     * @param [in] os_api: The Windows-specific OS API.
     */
    WindowsAffinity(const os_api& os_api);
    /**
     * @brief: Returns the number of cores.
     *
     * @return: The number of cores.
     */
    size_t count_cores() const;

protected:
    std::unique_ptr<database> m_database;
    const os_api& m_os_api;
};
/**
 * @brief: Windows-specific implementation of the CPU affinity management.
 */
class OsSpecificAffinity : public WindowsAffinity
{
public:
    /**
     * @brief: Constructor.
     *
     * @param [in] os_api: The Windows-specific OS API.
     */
    OsSpecificAffinity(const os_api& os_api) : WindowsAffinity{os_api} {}
};

} // namespace services
} // namespace dev_kit
} // namespace rivermax

#endif /* RDK_SERVICES_CPU_AFFINITY_WINDOWS_RIVERMAX_OS_AFFINITY_H_ */
