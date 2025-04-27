# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(parent_directory ${PROJECT_SOURCE_DIR})

add_library(rmax-util-cli INTERFACE)
target_sources(rmax-util-cli
    INTERFACE
        App.hpp
        CLI.hpp
        ConfigFwd.hpp
        Config.hpp
        Error.hpp
        FormatterFwd.hpp
        Formatter.hpp
        Macros.hpp
        Option.hpp
        Split.hpp
        StringTools.hpp
        Timer.hpp
        TypeTools.hpp
        Validators.hpp
        Version.hpp
)
target_include_directories(rmax-util-cli INTERFACE ${parent_directory})
add_library(CLI::CLI ALIAS rmax-util-cli)