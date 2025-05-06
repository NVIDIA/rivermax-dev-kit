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

#ifndef RDK_RIVERMAX_DEV_KIT_H_
#define RDK_RIVERMAX_DEV_KIT_H_

/**
 * @brief: Main API of Rivermax Dev Kit.
 *
 * The Rivermax Dev Kit provides a comprehensive set of APIs for developing
 * high-performance applications using the NVIDIA Rivermax library.
 *
 * ## Module Organization and Namespace Structure
 * The kit is organized into four primary modules, each with its corresponding namespace:
 * - Apps: Application-level functionality (@ref rivermax::dev_kit::apps)
 * - IO Node: Input/output management (@ref rivermax::dev_kit::io_node)
 * - Services: Supporting services (@ref rivermax::dev_kit::services)
 * - Core: Essential C++ Rivermax components and interfaces (@ref rivermax::dev_kit::core)
 *
 * All modules are contained within the main "rivermax" namespace, which contains the "dev_kit" namespace.
 *
 * ## Navigation
 * - Module headers are located in their respective include directories
 *       - Example: core/include/rdk/core/core.h
 * - Interface files are in subdirectories of their module
 *       - Example: core/stream/send/include/rdk/core/stream/send/media_stream.h
 *
 * ## File Structure
 * rivermax-dev-kit
 * |-- source
 * | |-- include/rdk
 * | | |-- facade.h
 * | | `--  rivermax_dev_kit.h
 * | |-- facade.cpp
 * | |-- apps
 * | | |-- include/rdk/apps/apps.h
 * | | ...
 * | |-- io_node
 * | | |-- include/rdk/io_node/io_node.h
 * | | ...
 * | |-- services
 * | | |-- include/rdk/services/services.h
 * | | |-- future_service_module_y
 * | | | |-- include/rdk/services/future_service_module_y.h
 * | | | ...
 * | | | |-- new_service_interface_a.h
 * | | | `--  new_service_interface_a.cpp
 * | | ...
 * | `--  core
 * | |-- include/rdk/core/core.h
 * | |-- future_core_module_x
 * | | |-- include/rdk/core/future_core_module_x.h
 * | | ...
 * | | |-- new_core_interface_a.h
 * | | `--  new_core_interface_a.cpp
 * | ...
 * |-- ...
 * `--  ...
 *
 * ## Example Usage
 * Rivermax Dev Kit offers flexible inclusion options to suit your development needs:
 *
 * 1. Include the entire API with a single header:
 *    ```
 *    #include "rdk/rivermax_dev_kit.h"
 *    ```
 *
 * 2. Include only specific modules you need, for example:
 *    ```
 *    #include "rdk/core/core.h"
 *    #include "rdk/services/services.h"
 *    ```
 *
 * 3. Include individual components for minimal dependencies:
 *    ```
 *    #include "rdk/core/stream/send/media_stream.h"
 *    ```
 *
 * Choose the approach that best balances convenience and compilation efficiency for your application.
 */

#include "rdk/apps/apps.h"
#include "rdk/io_node/io_node.h"
#include "rdk/services/services.h"
#include "rdk/core/core.h"

#endif /* RDK_RIVERMAX_DEV_KIT_H_ */
